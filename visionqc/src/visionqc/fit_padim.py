"""Fit PaDiM on normal images only.

RUN:
    python -m visionqc.fit_padim --config configs/synthetic.yaml

WHY THIS IS "FIT" AND NOT "TRAIN"
---------------------------------
There is no loss function here, no optimizer, no epochs. Nothing is learned by
gradient descent. We make a single pass over the normal images, accumulate the
mean and covariance of the feature vectors at each patch position, and we are
done.

That is a genuine practical advantage worth stating in an interview: on the
synthetic dataset this finishes in under a minute on a laptop CPU, versus tens
of minutes for the autoencoder on a GPU. When a factory adds a new part variant,
"re-fit in 60 seconds from 200 photos" is a very different operational story
from "retrain overnight".

NO AUGMENTATION HERE -- AND THIS IS DELIBERATE
----------------------------------------------
Note `train=False` on the loader below, unlike the autoencoder script.

PaDiM estimates a *distribution* of normal features per patch position. Random
rotation would scramble which patch is which: the feature at grid cell (3, 7)
would come from the part's rim in one image and its centre in the next. The
per-position Gaussians would then blur into one meaningless global distribution,
and you would throw away exactly the spatial specificity that makes PaDiM work.

This is a good example of "the right augmentation depends on what the model
assumes". The same rotation that helps the classifier generalise actively
destroys PaDiM. If your parts genuinely arrive at arbitrary rotation, the fix is
to align them upstream (or fit on a rotation-augmented set *and* accept a
looser, more permissive normal model), not to bolt augmentation onto PaDiM.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .config import Config
from .data.datasets import InspectionDataset, make_loader
from .models.padim import build_padim
from .utils import get_device, get_logger, load_json, set_seed, timer

logger = get_logger("visionqc.fit_padim")


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit PaDiM on normal images.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--splits", default=None)
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_json(Path(args.splits) if args.splits else run_dir / "splits.json")
    records = manifest["splits"]["anomaly_fit"]

    contaminated = [r["path"] for r in records if r["is_defect"]]
    if contaminated:
        raise AssertionError(
            f"{len(contaminated)} defective images in anomaly_fit. PaDiM models "
            "the distribution of NORMAL patches; a defect in the fit set widens "
            "that distribution to include defects, and detection silently fails."
        )

    d = cfg.data
    loader = make_loader(
        InspectionDataset(records, manifest["class_to_idx"], d.image_size,
                          d.mean, d.std, train=False),   # see docstring
        cfg.autoencoder.batch_size, shuffle=False, num_workers=d.num_workers,
    )
    logger.info("Fitting PaDiM on %d normal images | device=%s", len(records), device)

    model = build_padim(cfg.padim).to(device).eval()
    with timer("PaDiM fit", logger):
        model.fit(loader, device, logger)

    ckpt = run_dir / "padim.pt"
    model.save(ckpt)
    size_mb = ckpt.stat().st_size / 1e6
    logger.info("Saved -> %s (%.1f MB, self-contained)", ckpt, size_mb)

    # Quick smoke check so a broken fit is caught here, not three scripts later.
    with torch.no_grad():
        batch = next(iter(loader))
        scores = model.image_score(batch["image"].to(device))
    logger.info("Sanity check on normal images -- scores min %.2f max %.2f mean %.2f",
                scores.min(), scores.max(), scores.mean())


if __name__ == "__main__":
    main()
