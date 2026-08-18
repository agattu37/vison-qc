"""PaDiM — Patch Distribution Modeling, implemented from scratch.

Reference: Defard et al., "PaDiM: a Patch Distribution Modeling Framework for
Anomaly Detection and Localization" (ICPR 2021).

WHY IMPLEMENT THIS BY HAND INSTEAD OF `pip install anomalib`
------------------------------------------------------------
Using the library is the right call in a job. For a portfolio project it is the
wrong call, for one blunt reason: in an interview you will be asked how it
works, and "I called a function" ends the conversation. The algorithm is about
120 lines. Writing it means you can draw the Mahalanobis distance on a
whiteboard, and that is a completely different conversation.

We still list `anomalib` in the docs as the production alternative, because
knowing when *not* to hand-roll is also a signal.

THE ALGORITHM, IN PLAIN ENGLISH
-------------------------------
1. Push a normal image through a frozen, pretrained ResNet. Grab the feature
   maps from three depths and stack them channel-wise. Every location on that
   grid is now a vector describing one patch of the image, combining fine
   texture (early layer) with coarse structure (late layer).
2. Do that for every normal training image. Now, for each of the ~1024 grid
   positions, you have a cloud of a few hundred vectors: "here is the
   distribution of what patch (7, 12) normally looks like."
3. Summarise each cloud with a Gaussian — a mean vector and a covariance matrix.
   Note this is *per position*. The top-left corner of a part is expected to
   look different from its centre, and PaDiM models that explicitly. This is why
   it beats methods that model the whole image with one distribution.
4. At test time, compute how far each patch is from its own position's Gaussian,
   measured in Mahalanobis distance. Far = anomalous.

WHY MAHALANOBIS AND NOT EUCLIDEAN DISTANCE
------------------------------------------
Euclidean distance treats every feature dimension as equally important and
assumes they are independent. They are not. Some feature channels vary wildly
across perfectly normal parts (lighting, exact texture phase); others barely
vary at all. A 2-unit deviation in a stable channel is alarming; the same
deviation in a noisy channel is nothing.

Mahalanobis distance divides by the observed spread, per direction:

    d(x) = sqrt( (x - mu)^T * Sigma^-1 * (x - mu) )

Geometrically: Euclidean asks "how far in pixels?", Mahalanobis asks "how many
standard deviations, accounting for how the dimensions co-vary?". That is the
right question when the dimensions have wildly different scales, which they
always do in CNN features.

WHY RANDOM DIMENSION SELECTION
------------------------------
Concatenating layer1+layer2+layer3 of ResNet18 gives 448 channels. A 448x448
covariance matrix per position, times 1024 positions, is ~840 MB and slow to
invert. The paper's finding is that randomly keeping 100 of the 448 dimensions
performs *as well as* PCA-selecting them and as well as keeping all 448 — while
being ~20x cheaper. Random beats PCA here because PCA keeps the directions of
greatest variance in *normal* data, which are not necessarily the directions
where anomalies show up.

NOTE ON "TRAINING"
------------------
PaDiM has no gradient descent. Nothing is learned by backpropagation; the
backbone stays frozen. Fitting is one pass over the normal images to accumulate
means and covariances. On CPU this takes minutes, not hours — a genuine
practical advantage worth mentioning.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, ResNet34_Weights, resnet18, resnet34
from torchvision.models.feature_extraction import create_feature_extractor

from .autoencoder import gaussian_blur_map

_PADIM_BACKBONES = {
    "resnet18": (resnet18, ResNet18_Weights.DEFAULT),
    "resnet34": (resnet34, ResNet34_Weights.DEFAULT),
}


class PaDiM(nn.Module):
    """Fit on normal images only; score anything."""

    def __init__(
        self,
        backbone: str = "resnet18",
        layers: tuple[str, ...] = ("layer1", "layer2", "layer3"),
        n_components: int = 100,
        embed_grid: int = 32,
        reg: float = 0.01,
        seed: int = 1337,
        smooth_sigma: float = 4.0,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        if backbone not in _PADIM_BACKBONES:
            raise ValueError(
                f"PaDiM backbone must be one of {sorted(_PADIM_BACKBONES)}; "
                f"got '{backbone}'"
            )
        ctor, weights = _PADIM_BACKBONES[backbone]
        # `pretrained=False` exists only for offline unit tests and CI, where the
        # weight CDN may be unreachable. Never use it for real results: PaDiM's
        # entire premise is a *meaningful* feature space, and random features
        # measurably underperform ImageNet ones.
        net = ctor(weights=weights if pretrained else None)
        self.pretrained = pretrained

        # create_feature_extractor traces the model and returns intermediate
        # activations by name. Cleaner than registering forward hooks by hand,
        # and it is the supported torchvision API for exactly this.
        self.features = create_feature_extractor(
            net, return_nodes={ln: ln for ln in layers}
        )
        # Frozen forever: PaDiM's premise is a fixed, general feature space.
        for p in self.features.parameters():
            p.requires_grad = False
        self.features.eval()

        self.backbone_name = backbone
        self.layers = tuple(layers)
        self.n_components = n_components
        self.embed_grid = embed_grid
        self.reg = reg
        self.seed = seed
        self.smooth_sigma = smooth_sigma

        # Statistics, populated by fit(). Registered as buffers so that
        # .to(device), .state_dict() and torch.save all handle them correctly
        # without us writing any custom serialisation code.
        self.register_buffer("mean", torch.empty(0), persistent=True)
        self.register_buffer("chol", torch.empty(0), persistent=True)
        self.register_buffer("idx", torch.empty(0, dtype=torch.long), persistent=True)
        self._fitted = False

    # ---- feature embedding ------------------------------------------------
    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) -> (B, d, P) patch embeddings, d = n_components.

        P = embed_grid^2 patch positions, flattened row-major.
        """
        feats = self.features(x)
        target = (self.embed_grid, self.embed_grid)
        resized = [
            F.interpolate(feats[name], size=target, mode="bilinear", align_corners=False)
            for name in self.layers
        ]
        emb = torch.cat(resized, dim=1)          # (B, C_total, G, G)
        emb = emb[:, self.idx]                    # random dimension subset
        return emb.flatten(2)                     # (B, d, P)

    def _init_idx(self, total_channels: int, device: torch.device) -> None:
        """Choose the random channel subset once, deterministically.

        Determinism matters: the subset is part of the model. Re-drawing it at
        load time would pair the wrong statistics with the wrong channels and
        silently produce nonsense scores.
        """
        g = torch.Generator(device="cpu").manual_seed(self.seed)
        d = min(self.n_components, total_channels)
        idx = torch.randperm(total_channels, generator=g)[:d]
        self.idx = idx.sort().values.to(device)
        self.n_components = d

    # ---- fitting ----------------------------------------------------------
    @torch.no_grad()
    def fit(self, loader, device: torch.device, logger=None) -> "PaDiM":
        """One streaming pass over normal images.

        We accumulate sum(x) and sum(x x^T) per position instead of storing all
        embeddings. Memory then depends on the patch grid and dimension, not on
        how many images you have — so this scales to a dataset that does not fit
        in RAM. That is a small design decision but a real one.
        """
        self.features.to(device).eval()

        n_seen = 0
        sum_x: torch.Tensor | None = None
        sum_xx: torch.Tensor | None = None

        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)

            if self.idx.numel() == 0:
                feats = self.features(x)
                total_c = sum(feats[n].shape[1] for n in self.layers)
                self._init_idx(total_c, device)
                if logger:
                    logger.info(
                        "PaDiM embedding: %d channels -> %d random dims, %dx%d grid",
                        total_c, self.n_components, self.embed_grid, self.embed_grid,
                    )

            emb = self.embed(x)                       # (B, d, P)
            emb = emb.permute(2, 0, 1).contiguous()   # (P, B, d)

            if sum_x is None:
                p, _, d = emb.shape
                sum_x = torch.zeros(p, d, dtype=torch.float64, device=device)
                sum_xx = torch.zeros(p, d, d, dtype=torch.float64, device=device)

            e64 = emb.double()  # float64: covariance sums lose precision in fp32
            sum_x += e64.sum(dim=1)
            sum_xx += torch.einsum("pbd,pbe->pde", e64, e64)
            n_seen += x.shape[0]

        if n_seen < 2:
            raise ValueError(
                f"PaDiM needs >= 2 normal images to estimate a covariance; got {n_seen}."
            )

        mean = sum_x / n_seen                                     # (P, d)
        # Unbiased sample covariance: E[xx^T] - n*mu*mu^T, over (n-1).
        cov = (sum_xx - n_seen * torch.einsum("pd,pe->pde", mean, mean)) / (n_seen - 1)

        # Ridge regularisation. Without it, when the number of images is not
        # comfortably larger than the dimension, the covariance is singular or
        # near-singular and the inverse explodes. This one line is the difference
        # between "works" and "produces inf". Scale the ridge to the average
        # variance so `reg` means the same thing on any feature scale.
        d = mean.shape[1]
        eye = torch.eye(d, dtype=torch.float64, device=device)
        avg_var = torch.diagonal(cov, dim1=-2, dim2=-1).mean()
        cov = cov + self.reg * avg_var * eye

        # Cholesky factor instead of an explicit inverse: more numerically
        # stable, and the Mahalanobis distance becomes a triangular solve.
        try:
            chol = torch.linalg.cholesky(cov)
        except Exception:
            # Rare fallback: bump the ridge until the matrix is positive definite.
            for extra in (1e-3, 1e-2, 1e-1, 1.0):
                try:
                    chol = torch.linalg.cholesky(cov + extra * avg_var * eye)
                    if logger:
                        logger.warning("Cholesky needed extra ridge %.0e", extra)
                    break
                except Exception:
                    continue
            else:
                raise RuntimeError(
                    "Covariance is not positive definite even after heavy "
                    "regularisation. Increase padim.reg or use more fit images."
                )

        self.mean = mean.float()
        self.chol = chol.float()
        self._fitted = True
        if logger:
            logger.info("PaDiM fitted on %d normal images (%d patches, d=%d)",
                        n_seen, mean.shape[0], d)
        return self

    # ---- scoring ----------------------------------------------------------
    @torch.no_grad()
    def anomaly_map(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) -> (B, H, W) Mahalanobis distance map at input resolution."""
        if not self.is_fitted:
            raise RuntimeError("PaDiM.fit() must be called before scoring.")

        emb = self.embed(x)                        # (B, d, P)
        b, d, p = emb.shape
        # mean is (P, d); unsqueeze to (P, d, 1) so it broadcasts across the
        # batch dimension of the permuted embedding.
        delta = emb.permute(2, 1, 0) - self.mean.unsqueeze(2)      # (P, d, B)

        # Solve L y = delta. Then  y^T y = delta^T (L L^T)^-1 delta, which is
        # exactly the squared Mahalanobis distance -- without ever forming an
        # inverse matrix.
        y = torch.linalg.solve_triangular(self.chol, delta, upper=False)
        dist = y.pow(2).sum(dim=1).sqrt()          # (P, B)
        dist = dist.T.reshape(b, 1, self.embed_grid, self.embed_grid)

        amap = F.interpolate(
            dist, size=x.shape[-2:], mode="bilinear", align_corners=False
        ).squeeze(1)
        return gaussian_blur_map(amap, self.smooth_sigma)

    @torch.no_grad()
    def image_score(self, x: torch.Tensor) -> torch.Tensor:
        """One scalar per image: the maximum of its anomaly map.

        Max, not mean. A defect is small — often under 1% of the pixels. Its
        contribution to a mean is diluted to nothing by the 99% of normal
        pixels, and image-level AUROC collapses. Max asks the right question:
        "is there *anywhere* that looks wrong?"
        """
        return self.anomaly_map(x).flatten(1).max(dim=1).values

    @property
    def is_fitted(self) -> bool:
        return self.mean.numel() > 0 and self.chol.numel() > 0

    # ---- persistence ------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """Write a self-contained checkpoint.

        We deliberately store the backbone weights too, not just the fitted
        statistics. Two reasons:

        1. **Correctness.** The Gaussians were estimated in one specific feature
           space. Pair them with even slightly different backbone weights and
           every distance is meaningless. Shipping them together makes that
           mistake impossible.
        2. **Deployment.** The serving container then needs no network access at
           startup. On a free hosting tier that cold-starts your app, waiting on
           a weight download is the difference between a 3-second and a
           90-second first request -- or a crash, if the CDN is unreachable.

        Cost: about 45 MB extra. Worth it.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "mean": self.mean.cpu(),
                "chol": self.chol.cpu(),
                "idx": self.idx.cpu(),
                "features": {k: v.cpu() for k, v in self.features.state_dict().items()},
                "config": {
                    "backbone": self.backbone_name,
                    "layers": list(self.layers),
                    "n_components": self.n_components,
                    "embed_grid": self.embed_grid,
                    "reg": self.reg,
                    "seed": self.seed,
                    "smooth_sigma": self.smooth_sigma,
                    "pretrained": self.pretrained,
                },
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: torch.device | str = "cpu") -> "PaDiM":
        # weights_only=False because we store a config dict alongside tensors.
        # Only ever load checkpoints you produced yourself.
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        c = ckpt["config"]
        has_backbone = "features" in ckpt
        model = cls(
            backbone=c["backbone"], layers=tuple(c["layers"]),
            n_components=c["n_components"], embed_grid=c["embed_grid"],
            reg=c["reg"], seed=c["seed"], smooth_sigma=c["smooth_sigma"],
            # If the checkpoint carries the weights we are about to overwrite
            # them, so skip the download entirely -- faster and works offline.
            pretrained=False if has_backbone else c.get("pretrained", True),
        )
        if has_backbone:
            model.features.load_state_dict(ckpt["features"])
        model.mean = ckpt["mean"]
        model.chol = ckpt["chol"]
        model.idx = ckpt["idx"]
        model._fitted = True
        return model.to(device).eval()


def build_padim(cfg) -> PaDiM:
    return PaDiM(
        backbone=cfg.backbone, layers=tuple(cfg.layers),
        n_components=cfg.n_components, embed_grid=cfg.embed_grid,
        reg=cfg.reg, seed=cfg.seed, smooth_sigma=cfg.smooth_sigma,
        pretrained=cfg.pretrained,
    )
