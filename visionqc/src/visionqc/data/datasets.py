"""Datasets and transforms.

WHY THIS FILE EXISTS
--------------------
Two ideas beginners usually get wrong, both fixed here:

**1. Augmentation belongs to the split, not the dataset.**
Training data gets randomly perturbed so the model cannot memorise; validation
and test data get a fixed, deterministic pipeline so your metric means the same
thing every time you compute it. Applying random augmentation at eval time is a
silent bug — your numbers wobble and you blame the model.

**2. Which augmentations are *valid* is a domain question, not a default.**
For a machined part on a conveyor:
  - rotation / flips  -> YES. The part arrives at arbitrary orientation.
  - brightness / contrast jitter -> YES. Factory lighting drifts, lamps age.
  - small translation -> YES. Part placement is not pixel-perfect.
  - heavy colour jitter (hue) -> NO. It would recolour a rust stain to look like
    clean metal, destroying the very signal we need.
  - random erasing / cutout -> NO. It punches synthetic holes that look exactly
    like the contamination defects we are trying to detect. You would be
    teaching the model that defects are normal.

That last point is worth saying in an interview verbatim. It shows you reasoned
about the domain instead of copy-pasting an augmentation recipe.

We use torchvision.transforms.v2, the current API. The older `transforms` module
still works but v2 is faster, supports masks/boxes natively, and is what new
code should use.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

# Pillow >=10 renamed the resampling enum; this is the current spelling.
_BILINEAR = v2.InterpolationMode.BILINEAR


def build_transforms(
    image_size: int,
    mean: Sequence[float],
    std: Sequence[float],
    train: bool,
) -> v2.Compose:
    """Return the v2 pipeline for a split.

    Note the ordering: geometry first, then photometry, then tensor conversion,
    then normalisation. Normalising before augmenting would mean the jitter
    operates on already-standardised values, which is not what the jitter
    parameters assume.
    """
    if train:
        steps = [
            v2.Resize((image_size, image_size), interpolation=_BILINEAR),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            # Full 360 deg: a part on a conveyor has no canonical "up".
            v2.RandomRotation(degrees=180, interpolation=_BILINEAR),
            v2.RandomAffine(degrees=0, translate=(0.04, 0.04), scale=(0.95, 1.05),
                            interpolation=_BILINEAR),
            # Brightness/contrast only. No hue shift -- see module docstring.
            v2.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.0, hue=0.0),
        ]
    else:
        steps = [v2.Resize((image_size, image_size), interpolation=_BILINEAR)]

    steps += [
        v2.ToImage(),                              # PIL -> tv_tensors.Image
        v2.ToDtype(torch.float32, scale=True),     # uint8 [0,255] -> float [0,1]
        v2.Normalize(mean=list(mean), std=list(std)),
    ]
    return v2.Compose(steps)


class InspectionDataset(Dataset):
    """Serves one split of the manifest.

    Returns a dict rather than a tuple. Tuples force you to remember positional
    order at every call site; a dict lets you add `mask` for the pixel-level
    evaluation without breaking anything that already unpacks the batch.
    """

    def __init__(
        self,
        records: list[dict[str, Any]],
        class_to_idx: dict[str, int],
        image_size: int = 256,
        mean: Sequence[float] = (0.485, 0.456, 0.406),
        std: Sequence[float] = (0.229, 0.224, 0.225),
        train: bool = False,
        load_masks: bool = False,
    ) -> None:
        if not records:
            raise ValueError("InspectionDataset received an empty record list.")
        self.records = records
        self.class_to_idx = class_to_idx
        self.image_size = image_size
        self.load_masks = load_masks
        self.transform = build_transforms(image_size, mean, std, train)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> dict[str, Any]:
        rec = self.records[i]
        # .convert("RGB") is not optional: a greyscale PNG loads as 1 channel and
        # would crash the first conv layer, which expects 3.
        img = Image.open(rec["path"]).convert("RGB")
        x = self.transform(img)

        item: dict[str, Any] = {
            "image": x,
            "label": torch.tensor(self.class_to_idx[rec["label"]], dtype=torch.long),
            "is_defect": torch.tensor(rec["is_defect"], dtype=torch.long),
            "path": rec["path"],
        }

        if self.load_masks:
            item["mask"] = self._load_mask(rec)
        return item

    def _load_mask(self, rec: dict[str, Any]) -> torch.Tensor:
        """Binary ground-truth mask at image_size, shape (H, W), values {0,1}.

        Good images get an all-zero mask so that pixel-level AUROC can include
        them — their pixels are all true negatives, which is exactly right.
        NEAREST resampling is mandatory: bilinear would invent grey values
        between 0 and 1 and blur the class boundary.
        """
        if not rec.get("mask"):
            return torch.zeros(self.image_size, self.image_size, dtype=torch.uint8)
        m = Image.open(rec["mask"]).convert("L").resize(
            (self.image_size, self.image_size), Image.NEAREST
        )
        arr = (np.array(m) > 127).astype(np.uint8)
        return torch.from_numpy(arr)


def make_loader(
    dataset: InspectionDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 2,
    drop_last: bool = False,
) -> DataLoader:
    """DataLoader with settings that behave the same on every machine.

    `persistent_workers` only makes sense when workers exist; passing it with
    num_workers=0 raises. `pin_memory` is a CUDA-only optimisation, so we gate it
    to avoid a warning on CPU-only boxes.
    """
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "drop_last": drop_last,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
    return DataLoader(dataset, **kwargs)


def compute_class_weights(
    records: list[dict[str, Any]], class_to_idx: dict[str, int]
) -> torch.Tensor:
    """Inverse-frequency weights, normalised to mean 1.

    Why: with 200 good images and 10 scratches, a model that always says "good"
    is 95% accurate and completely useless. Weighting the loss by 1/frequency
    makes one scratch mistake cost as much as twenty good-image mistakes.

    Normalising to mean 1 keeps the overall loss magnitude comparable to the
    unweighted case, so your learning rate does not silently need retuning.
    """
    counts = np.zeros(len(class_to_idx), dtype=np.float64)
    for r in records:
        counts[class_to_idx[r["label"]]] += 1
    # Guard against a class with zero examples -> avoid division by zero.
    counts = np.maximum(counts, 1.0)
    w = counts.sum() / (len(counts) * counts)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)
