"""The inference engine: load artifacts once, score images.

WHY A SHARED ENGINE INSTEAD OF SCORING CODE IN EACH SCRIPT
-----------------------------------------------------------
This is the single most important structural decision in the repo, and it is
worth being able to articulate.

If `evaluate.py` and `api/main.py` each implement their own preprocessing and
scoring, they *will* drift. Someone changes the resize interpolation in one
place, or forgets the normalisation, and now the API returns different scores
than the ones you validated and published. That class of bug is called
training/serving skew, it is silent, and it is one of the most common ways real
ML systems fail in production.

The fix is boring and effective: exactly one code path turns an image into a
score. Evaluation calls it. The API calls it. If they ever disagree, it is a
bug in this file, not a mystery.

The engine loads everything from disk artifacts -- checkpoints plus a
thresholds JSON -- so serving has no dependency on the training code beyond the
model class definitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .data.datasets import build_transforms
from .decision import Decision, Thresholds, decide
from .explain.gradcam import GradCAM
from .explain.overlay import denormalise, overlay_heatmap, side_by_side
from .models.autoencoder import ConvAutoencoder, gaussian_blur_map
from .models.classifier import DefectClassifier
from .models.padim import PaDiM
from .utils import get_logger

logger = get_logger("visionqc.inference")


@dataclass
class ScoreResult:
    """Everything one image produced. Maps are kept as tensors so callers can
    decide whether to render them -- rendering is expensive and evaluation does
    not need it for most images."""

    defect_probability: float | None = None
    predicted_class: str | None = None
    class_confidence: float | None = None
    class_probabilities: dict[str, float] | None = None
    anomaly_score: float | None = None
    anomaly_map: torch.Tensor | None = None      # (H, W)
    gradcam_map: torch.Tensor | None = None      # (H, W)
    image_tensor: torch.Tensor | None = None     # (3, H, W) normalised


class InspectionEngine:
    """Loads whichever artifacts exist and scores images with them.

    Partial loading is intentional: if you have only fitted PaDiM and have not
    trained a classifier yet, the engine works with what it has. Requiring all
    artifacts would make the project un-demoable until the very last step.
    """

    def __init__(
        self,
        run_dir: str | Path,
        device: torch.device | str = "cpu",
        anomaly_backend: str = "padim",
    ) -> None:
        self.run_dir = Path(run_dir)
        self.device = torch.device(device)
        self.anomaly_backend = anomaly_backend

        self.classifier: DefectClassifier | None = None
        self.classes: list[str] = []
        self.padim: PaDiM | None = None
        self.autoencoder: ConvAutoencoder | None = None
        self.ae_smooth_sigma: float = 4.0

        self.image_size: int = 256
        self.mean: tuple[float, ...] = (0.485, 0.456, 0.406)
        self.std: tuple[float, ...] = (0.229, 0.224, 0.225)

        self.thresholds = Thresholds()
        self._declared_sizes: list[int] = []
        self._load()
        # Built once, reused for every request. Rebuilding a transform pipeline
        # per request is pure overhead on a latency budget.
        self.transform = build_transforms(self.image_size, self.mean, self.std, train=False)

    # ---- loading ----------------------------------------------------------
    def _load(self) -> None:
        loaded: list[str] = []

        clf_path = self.run_dir / "classifier.pt"
        if clf_path.exists():
            ck = torch.load(clf_path, map_location="cpu", weights_only=False)
            self.classes = ck["classes"]
            model = DefectClassifier(
                num_classes=len(self.classes), backbone=ck["backbone"],
                # The state dict overwrites every weight, so downloading
                # ImageNet weights first would be wasted time and bandwidth.
                pretrained=False,
            )
            model.load_state_dict(ck["state_dict"])
            self.classifier = model.to(self.device).eval()
            self.image_size = ck.get("image_size", self.image_size)
            self.mean = tuple(ck.get("mean", self.mean))
            self.std = tuple(ck.get("std", self.std))
            self._declared_sizes.append(self.image_size)
            loaded.append(f"classifier({ck['backbone']}, {len(self.classes)} classes)")

        padim_path = self.run_dir / "padim.pt"
        if padim_path.exists():
            self.padim = PaDiM.load(padim_path, self.device)
            loaded.append("padim")

        ae_path = self.run_dir / "autoencoder.pt"
        if ae_path.exists():
            ck = torch.load(ae_path, map_location="cpu", weights_only=False)
            ae = ConvAutoencoder(ck["base_channels"], ck["latent_channels"])
            ae.load_state_dict(ck["state_dict"])
            self.autoencoder = ae.to(self.device).eval()
            self.ae_smooth_sigma = ck.get("smooth_sigma", 4.0)
            self.image_size = ck.get("image_size", self.image_size)
            self._declared_sizes.append(self.image_size)
            loaded.append("autoencoder")

        thr_path = self.run_dir / "thresholds.json"
        if thr_path.exists():
            import json
            with open(thr_path, "r", encoding="utf-8") as fh:
                self.thresholds = Thresholds.from_dict(json.load(fh))
            loaded.append("thresholds")

        # Every model in a run must agree on input size and normalisation.
        # Without this check, loading a classifier trained at 224 alongside an
        # autoencoder trained at 256 silently preprocesses one of them wrongly
        # -- scores stay plausible, results are quietly garbage. Fail loudly.
        if len(set(self._declared_sizes)) > 1:
            raise ValueError(
                f"Checkpoints in {self.run_dir} disagree on image_size: "
                f"{sorted(set(self._declared_sizes))}. Retrain them from the same "
                "config, or point --run-dir at a consistent run."
            )

        if not loaded:
            raise FileNotFoundError(
                f"No model artifacts found in {self.run_dir}. Train something "
                f"first, e.g.:\n  python -m visionqc.fit_padim --config <cfg>"
            )
        logger.info("Engine loaded: %s", ", ".join(loaded))

    @property
    def has_classifier(self) -> bool:
        return self.classifier is not None

    @property
    def has_anomaly(self) -> bool:
        return self._anomaly_model() is not None

    def _anomaly_model(self):
        if self.anomaly_backend == "padim":
            return self.padim
        if self.anomaly_backend == "autoencoder":
            return self.autoencoder
        raise ValueError(f"Unknown anomaly_backend '{self.anomaly_backend}'")

    # ---- preprocessing ----------------------------------------------------
    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """PIL image -> (1, 3, S, S) normalised tensor on the engine's device."""
        return self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)

    # ---- scoring ----------------------------------------------------------
    @torch.no_grad()
    def _anomaly_batch(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (scores (B,), maps (B,H,W)) from whichever backend is active."""
        model = self._anomaly_model()
        if model is None:
            raise RuntimeError("No anomaly model loaded.")
        if isinstance(model, PaDiM):
            amap = model.anomaly_map(x)
        else:
            amap = gaussian_blur_map(model.anomaly_map(x), self.ae_smooth_sigma)
        return amap.flatten(1).max(dim=1).values, amap

    def score_batch(
        self, x: torch.Tensor, want_maps: bool = False, want_gradcam: bool = False
    ) -> list[ScoreResult]:
        """Score a batch. This is the hot path used by evaluation."""
        results = [ScoreResult() for _ in range(x.shape[0])]

        if self.has_classifier:
            with torch.no_grad():
                probs = F.softmax(self.classifier(x), dim=1)
            conf, pred = probs.max(dim=1)
            # Class 0 is 'good' by construction of the split manifest, so the
            # binary defect probability falls straight out of the softmax.
            # One model, two answers.
            p_defect = 1.0 - probs[:, 0]
            for i, r in enumerate(results):
                r.defect_probability = float(p_defect[i])
                r.predicted_class = self.classes[int(pred[i])]
                r.class_confidence = float(conf[i])
                r.class_probabilities = {
                    c: float(probs[i, j]) for j, c in enumerate(self.classes)
                }

        if self.has_anomaly:
            scores, maps = self._anomaly_batch(x)
            for i, r in enumerate(results):
                r.anomaly_score = float(scores[i])
                if want_maps:
                    r.anomaly_map = maps[i].detach().cpu()

        if want_gradcam and self.has_classifier:
            with GradCAM(self.classifier, self.classifier.target_layer()) as cam:
                heat, _ = cam(x)
            for i, r in enumerate(results):
                r.gradcam_map = heat[i].detach().cpu()

        if want_maps:
            for i, r in enumerate(results):
                r.image_tensor = x[i].detach().cpu()
        return results

    def score_image(
        self, image: Image.Image, want_maps: bool = True, want_gradcam: bool = True
    ) -> ScoreResult:
        x = self.preprocess(image)
        return self.score_batch(x, want_maps, want_gradcam)[0]

    # ---- decision ---------------------------------------------------------
    def decide(self, r: ScoreResult, primary: str = "fusion") -> Decision:
        return decide(
            thresholds=self.thresholds,
            defect_probability=r.defect_probability,
            predicted_class=r.predicted_class,
            class_confidence=r.class_confidence,
            anomaly_score=r.anomaly_score,
            primary=primary,
        )

    def inspect(self, image: Image.Image, primary: str = "fusion") -> tuple[Decision, ScoreResult]:
        """The full pipeline for one image -- what the API endpoint calls."""
        r = self.score_image(image, want_maps=True, want_gradcam=self.has_classifier)
        return self.decide(r, primary), r

    # ---- visualisation ----------------------------------------------------
    def render_panel(self, r: ScoreResult) -> np.ndarray:
        """original | anomaly heatmap | Grad-CAM -- the demo image.

        Whichever maps are missing are simply omitted, so this works with a
        partially-trained project.
        """
        if r.image_tensor is None:
            raise ValueError("render_panel needs want_maps=True at scoring time.")
        base = denormalise(r.image_tensor, self.mean, self.std)
        panels = [base]
        if r.anomaly_map is not None:
            panels.append(overlay_heatmap(base, r.anomaly_map))
        if r.gradcam_map is not None:
            panels.append(overlay_heatmap(base, r.gradcam_map, cmap="viridis"))
        return side_by_side(panels)
