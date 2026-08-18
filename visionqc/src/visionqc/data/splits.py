"""Build a deterministic, leak-free split manifest.

WHY THIS FILE EXISTS — read this part carefully, it is the most interview-relevant
idea in the whole data layer.
---------------------------------------------------------------------------
Our two models have *incompatible* training requirements:

* The **anomaly detector** must see only normal images during fitting. That is
  the entire premise: it learns "what normal looks like" and flags deviation.
* The **classifier** must see labelled defects during training. It cannot learn
  a class it has never seen.

The naive approach is to give each model its own split. That is a trap: you then
have no way to compare them, because they were scored on different test sets.
Any difference in their numbers could just be a difference in test difficulty.

So we impose one rule: **there is exactly one test set, and neither model sees
any part of it, ever.**

    ┌──────────────────────────────────────────────────────────┐
    │ train/good/          (normal images, no defects present)  │
    │   └─> anomaly_fit    100% -- fits PaDiM / trains the AE   │
    │   └─> sup_train/val  also usable as labelled "good"       │
    ├──────────────────────────────────────────────────────────┤
    │ test/good/ + test/<defect>/   (the only labelled defects)  │
    │   └─> HOLDOUT TEST   50%, stratified -- untouched         │
    │   └─> sup_train      ~40%  -- classifier learns from here │
    │   └─> sup_val        ~10%  -- early stopping, thresholds  │
    └──────────────────────────────────────────────────────────┘

Two consequences worth stating out loud in an interview:

1. The classifier is trained on a *deliberately small* number of defects,
   because that is the real constraint the PRD describes. We are not pretending
   defects are plentiful.
2. Thresholds are chosen on **sup_val**, never on the test set. Choosing an
   operating point on your test set is a subtle but real form of leakage, and it
   is the single most common flaw in portfolio projects.

Stratification: we sample the holdout per class, not globally. With ~25 images
of a rare defect type, a global random split can easily give you zero of them in
test, and then your recall for that class is undefined.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ..utils import get_logger, save_json

logger = get_logger("visionqc.splits")

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
GOOD_LABEL = "good"


def _list_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def scan_dataset(root: str | Path) -> dict[str, Any]:
    """Discover an MVTec-style tree. Returns raw pools before splitting."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(
            f"Dataset root not found: {root}\n"
            f"Run: python -m visionqc.data.synthetic --root {root}"
        )

    train_good = _list_images(root / "train" / "good")
    test_dir = root / "test"
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Missing {test_dir}. Expected MVTec-style layout.")

    test_pools: dict[str, list[Path]] = {}
    for sub in sorted(p for p in test_dir.iterdir() if p.is_dir()):
        imgs = _list_images(sub)
        if imgs:
            test_pools[sub.name] = imgs

    if GOOD_LABEL not in test_pools:
        raise ValueError(
            f"No 'good' folder under {test_dir}. The test set must contain normal "
            "samples or AUROC is undefined (you cannot rank without negatives)."
        )

    defect_types = sorted(k for k in test_pools if k != GOOD_LABEL)
    if not defect_types:
        raise ValueError(f"No defect folders under {test_dir}.")

    logger.info("Scanned %s", root)
    logger.info("  train/good: %d images", len(train_good))
    for k in sorted(test_pools):
        logger.info("  test/%-16s %d images", k + ":", len(test_pools[k]))

    return {"root": str(root), "train_good": train_good,
            "test_pools": test_pools, "defect_types": defect_types}


def _mask_for(root: Path, image_path: Path, label: str) -> str | None:
    """Locate the MVTec ground-truth mask for a defective image, if present.

    MVTec convention: test/<defect>/000.png -> ground_truth/<defect>/000_mask.png
    Good images have no mask (their mask is implicitly all-zero).
    """
    if label == GOOD_LABEL:
        return None
    gt_dir = root / "ground_truth" / label
    if not gt_dir.is_dir():
        return None
    stem = image_path.stem
    for candidate in (f"{stem}_mask.png", f"{stem}.png", f"{stem}_mask.bmp"):
        p = gt_dir / candidate
        if p.exists():
            return str(p)
    return None


def build_splits(
    root: str | Path,
    holdout_frac: float = 0.5,
    val_frac: float = 0.2,
    seed: int = 42,
) -> dict[str, Any]:
    """Create the split manifest described in this module's docstring."""
    if not 0.0 < holdout_frac < 1.0:
        raise ValueError("holdout_frac must be in (0, 1)")
    if not 0.0 <= val_frac < 1.0:
        raise ValueError("val_frac must be in [0, 1)")

    info = scan_dataset(root)
    root_p = Path(info["root"])
    rng = np.random.default_rng(seed)

    holdout: list[dict] = []
    remainder: list[dict] = []

    # --- stratified holdout: independently per class -----------------------
    for label, paths in sorted(info["test_pools"].items()):
        idx = rng.permutation(len(paths))
        n_hold = max(1, int(round(len(paths) * holdout_frac)))
        # Guarantee at least one image survives for the supervised pool too,
        # otherwise a class can vanish from training entirely.
        n_hold = min(n_hold, len(paths) - 1) if len(paths) > 1 else len(paths)
        for rank, i in enumerate(idx):
            p = paths[i]
            record = {
                "path": str(p),
                "label": label,
                "is_defect": int(label != GOOD_LABEL),
                "mask": _mask_for(root_p, p, label),
                "source": "test",
            }
            (holdout if rank < n_hold else remainder).append(record)

    # --- train/good: all of it fits the anomaly model ----------------------
    anomaly_fit = [
        {"path": str(p), "label": GOOD_LABEL, "is_defect": 0, "mask": None,
         "source": "train"}
        for p in info["train_good"]
    ]

    # --- supervised pool = leftover labelled test data + all train/good ----
    sup_pool = remainder + [dict(r) for r in anomaly_fit]

    # Stratified train/val split over the supervised pool.
    by_label: dict[str, list[dict]] = defaultdict(list)
    for r in sup_pool:
        by_label[r["label"]].append(r)

    sup_train: list[dict] = []
    sup_val: list[dict] = []
    for label, records in sorted(by_label.items()):
        order = rng.permutation(len(records))
        n_val = int(round(len(records) * val_frac))
        # Keep at least one val sample per class when the class allows it, so
        # per-class validation metrics are always defined.
        if len(records) > 1:
            n_val = max(1, min(n_val, len(records) - 1))
        for rank, i in enumerate(order):
            (sup_val if rank < n_val else sup_train).append(records[i])

    classes = [GOOD_LABEL] + list(info["defect_types"])

    manifest = {
        "root": str(root_p),
        "seed": seed,
        "holdout_frac": holdout_frac,
        "val_frac": val_frac,
        "classes": classes,
        "class_to_idx": {c: i for i, c in enumerate(classes)},
        "splits": {
            "anomaly_fit": anomaly_fit,
            "sup_train": sup_train,
            "sup_val": sup_val,
            "test": holdout,
        },
    }
    _verify_no_leak(manifest)
    _log_summary(manifest)
    return manifest


def _verify_no_leak(manifest: dict[str, Any]) -> None:
    """Assert the test set shares no image with any training split.

    This runs on every build. A leak check that only runs when you remember to
    run it is not a leak check.
    """
    splits = manifest["splits"]
    test_paths = {r["path"] for r in splits["test"]}
    for name in ("anomaly_fit", "sup_train", "sup_val"):
        overlap = test_paths & {r["path"] for r in splits[name]}
        if overlap:
            raise AssertionError(
                f"LEAK: {len(overlap)} images appear in both 'test' and '{name}'. "
                f"Example: {sorted(overlap)[0]}"
            )
    # The anomaly model must never have seen a defect.
    bad = [r["path"] for r in splits["anomaly_fit"] if r["is_defect"]]
    if bad:
        raise AssertionError(
            f"LEAK: {len(bad)} defective images in 'anomaly_fit'. The unsupervised "
            "path is only meaningful if it is fitted on normal data alone."
        )
    logger.info("Leak check passed: test set is disjoint from all training splits.")


def _log_summary(manifest: dict[str, Any]) -> None:
    for name, records in manifest["splits"].items():
        counts: dict[str, int] = defaultdict(int)
        for r in records:
            counts[r["label"]] += 1
        detail = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        logger.info("%-12s n=%-5d [%s]", name, len(records), detail)
    n_masks = sum(1 for r in manifest["splits"]["test"] if r["mask"])
    logger.info("Test images with ground-truth masks: %d", n_masks)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the VisionQC split manifest.")
    ap.add_argument("--root", default="data/synthetic")
    ap.add_argument("--out", default="artifacts/splits.json")
    ap.add_argument("--holdout-frac", type=float, default=0.5)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    manifest = build_splits(args.root, args.holdout_frac, args.val_frac, args.seed)
    save_json(manifest, args.out)
    logger.info("Wrote manifest -> %s", args.out)


if __name__ == "__main__":
    main()
