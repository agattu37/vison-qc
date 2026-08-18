"""Train the convolutional autoencoder on normal images only.

RUN:
    python -m visionqc.train_autoencoder --config configs/synthetic.yaml

THE ONE RULE THAT MATTERS
-------------------------
This script must never see a defective image. Not one. The manifest builder
already asserts that `anomaly_fit` contains no defects, and we re-assert it here
before the first batch. Defence in depth: a data leak here would not crash, it
would just quietly produce a model that reconstructs defects nicely and detects
nothing -- and you would spend a day wondering why your AUROC is 0.5.

WHY WE HOLD OUT SOME NORMAL IMAGES
----------------------------------
We split `anomaly_fit` into fit/val (both normal). The validation reconstruction
loss tells us when the autoencoder stops improving at its actual job. We cannot
use defect data for early stopping -- that would be exactly the supervision we
are pretending not to have, and it would make the "unsupervised" claim false.

A SUBTLETY WORTH KNOWING
------------------------
Lower reconstruction loss is not automatically better for anomaly detection.
Train long enough with enough capacity and the autoencoder starts reconstructing
*anything*, including defects, and detection performance falls even as the loss
keeps dropping. This is the "identity function" failure. If your AUROC gets
worse as training continues, that is the cause -- shrink `latent_channels` or
stop earlier. We log both so the effect is visible rather than mysterious.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .config import Config
from .data.datasets import InspectionDataset, make_loader
from .models.autoencoder import build_autoencoder
from .utils import count_parameters, get_device, get_logger, load_json, save_json, set_seed

logger = get_logger("visionqc.train_ae")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the VisionQC autoencoder.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--splits", default=None)
    ap.add_argument("--val-frac", type=float, default=0.1,
                    help="fraction of NORMAL images held out for early stopping")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_json(Path(args.splits) if args.splits else run_dir / "splits.json")
    records = manifest["splits"]["anomaly_fit"]

    # Defence in depth -- see module docstring.
    contaminated = [r["path"] for r in records if r["is_defect"]]
    if contaminated:
        raise AssertionError(
            f"{len(contaminated)} defective images found in anomaly_fit. "
            "The unsupervised path must train on normal data only."
        )

    rng = np.random.default_rng(cfg.seed)
    order = rng.permutation(len(records))
    n_val = max(1, int(len(records) * args.val_frac))
    val_recs = [records[i] for i in order[:n_val]]
    fit_recs = [records[i] for i in order[n_val:]]

    d = cfg.data
    c2i = manifest["class_to_idx"]
    # NOTE train=True on the fit split: augmentation still helps here. It widens
    # the model's notion of "normal" to include the lighting and orientation
    # variation a real line produces, so a legitimately rotated part does not
    # later get flagged as an anomaly.
    fit_loader = make_loader(
        InspectionDataset(fit_recs, c2i, d.image_size, d.mean, d.std, train=True),
        cfg.autoencoder.batch_size, True, d.num_workers,
    )
    val_loader = make_loader(
        InspectionDataset(val_recs, c2i, d.image_size, d.mean, d.std, train=False),
        cfg.autoencoder.batch_size, False, d.num_workers,
    )
    logger.info("Fit on %d normal images | val on %d normal images",
                len(fit_recs), len(val_recs))

    model = build_autoencoder(cfg.autoencoder).to(device)
    logger.info("Autoencoder: %.2fM params | latent %d x %d x %d",
                count_parameters(model)[0] / 1e6, cfg.autoencoder.latent_channels,
                d.image_size // 16, d.image_size // 16)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.autoencoder.lr,
        weight_decay=cfg.autoencoder.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, cfg.autoencoder.epochs)
    )

    best_val, best_state, best_epoch = float("inf"), None, -1
    history: list[dict] = []

    for epoch in range(cfg.autoencoder.epochs):
        model.train()
        tr_sum, tr_n = 0.0, 0
        for batch in fit_loader:
            x = batch["image"].to(device, non_blocking=True)
            recon = model(x)
            loss = criterion(recon, x)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            tr_sum += loss.item() * x.size(0)
            tr_n += x.size(0)

        model.eval()
        va_sum, va_n = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["image"].to(device, non_blocking=True)
                va_sum += criterion(model(x), x).item() * x.size(0)
                va_n += x.size(0)

        scheduler.step()
        tr_loss, va_loss = tr_sum / max(tr_n, 1), va_sum / max(va_n, 1)
        history.append({"epoch": epoch, "train_mse": tr_loss, "val_mse": va_loss,
                        "lr": optimizer.param_groups[0]["lr"]})
        logger.info("epoch %02d | train MSE %.5f | val MSE %.5f", epoch, tr_loss, va_loss)

        if va_loss < best_val:
            best_val, best_epoch = va_loss, epoch
            best_state = copy.deepcopy(
                {k: v.detach().cpu() for k, v in model.state_dict().items()}
            )

    if best_state is None:
        raise RuntimeError("No checkpoint produced -- is anomaly_fit empty?")

    ckpt_path = run_dir / "autoencoder.pt"
    torch.save(
        {
            "state_dict": best_state,
            "base_channels": cfg.autoencoder.base_channels,
            "latent_channels": cfg.autoencoder.latent_channels,
            "smooth_sigma": cfg.autoencoder.smooth_sigma,
            "image_size": d.image_size,
            "mean": list(d.mean), "std": list(d.std),
            "best_epoch": best_epoch, "best_val_mse": best_val,
        },
        ckpt_path,
    )
    save_json(history, run_dir / "autoencoder_history.json")
    logger.info("Best epoch %d | val MSE %.5f | saved -> %s",
                best_epoch, best_val, ckpt_path)


if __name__ == "__main__":
    main()
