"""Generate a synthetic, MVTec-style inspection dataset.

WHY THIS FILE EXISTS
--------------------
MVTec AD requires a licence click-through and a ~5 GB download. Kaggle needs an
API token. Neither is hard, but both mean you cannot run a single line of your
own pipeline on day one — and a pipeline you have not run is a pipeline that
does not work.

So we ship a generator that fabricates the *exact folder layout* MVTec uses:
brushed-metal discs that are mostly fine, plus four defect types with
pixel-perfect ground-truth masks. You can train, evaluate, and serve within
minutes, then point the same code at the real dataset by changing one path.

This is a real engineering habit, not a shortcut. Building a synthetic fixture
that matches your production schema is how you test data pipelines without
waiting on data, and it is a good thing to be able to say you did.

HONEST LIMITATION (say this in interviews, do not hide it)
----------------------------------------------------------
Synthetic defects are *easier* than real ones: the noise model is known, and
lighting variation is simpler than a real factory. Numbers on synthetic data
will be optimistic. Use it to prove the code path works; report MVTec numbers
as your actual result.

Layout produced (identical to MVTec AD):

    <root>/
        train/good/000.png              <- normal only, this is what fits the AD model
        test/good/000.png               <- normal test samples
        test/scratch/000.png            <- defective test samples, by type
        test/dent/000.png
        test/contamination/000.png
        test/crack/000.png
        ground_truth/scratch/000_mask.png   <- binary mask, 255 = defect
        ...
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

DEFECT_TYPES = ("scratch", "dent", "contamination", "crack")


# --------------------------------------------------------------------------
# Base part rendering
# --------------------------------------------------------------------------
def _brushed_texture(size: int, rng: np.random.Generator) -> np.ndarray:
    """Anisotropic noise that reads as machined/brushed metal.

    Trick: blur white noise *much* more along x than y. That directional blur is
    what makes streaks instead of blobs.
    """
    noise = rng.standard_normal((size, size))
    streaks = gaussian_filter(noise, sigma=(0.6, 9.0))
    grain = gaussian_filter(rng.standard_normal((size, size)), sigma=1.1)
    tex = 0.75 * streaks / (streaks.std() + 1e-8) + 0.25 * grain
    return tex


def _radial_mask(size: int, cx: float, cy: float, radius: float) -> np.ndarray:
    """Soft-edged disc mask in [0, 1]."""
    yy, xx = np.mgrid[0:size, 0:size]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    # Smooth 2-pixel falloff avoids a jagged, obviously-synthetic edge.
    return np.clip((radius - dist) / 2.0, 0.0, 1.0)


def _render_good_part(size: int, rng: np.random.Generator) -> np.ndarray:
    """One clean part image as float32 in [0, 1], shape (size, size)."""
    # Part is roughly centred but jitters, like a part on a moving conveyor.
    cx = size / 2 + rng.uniform(-size * 0.02, size * 0.02)
    cy = size / 2 + rng.uniform(-size * 0.02, size * 0.02)
    radius = size * rng.uniform(0.36, 0.40)

    disc = _radial_mask(size, cx, cy, radius)
    inner = _radial_mask(size, cx, cy, radius * 0.30)  # central bore

    base = 0.16 + 0.03 * rng.standard_normal()  # dark background tray
    body = 0.62 + 0.05 * rng.standard_normal()  # part brightness varies per unit

    img = np.full((size, size), base, dtype=np.float32)
    tex = _brushed_texture(size, rng)
    img = img + disc * (body - base + 0.05 * tex)

    # Machined rim: slightly brighter ring near the outer edge.
    rim = _radial_mask(size, cx, cy, radius) - _radial_mask(size, cx, cy, radius * 0.93)
    img += 0.10 * np.clip(rim, 0, 1)

    # Central bore is darker.
    img -= 0.34 * inner

    # Non-uniform illumination: a broad light gradient, as in a real light tent.
    yy, xx = np.mgrid[0:size, 0:size]
    ang = rng.uniform(0, 2 * np.pi)
    grad = (np.cos(ang) * xx + np.sin(ang) * yy) / size
    img += rng.uniform(0.02, 0.07) * grad

    # Sensor noise last, so it is not smoothed away by anything above.
    img += rng.normal(0, 0.012, (size, size))
    return np.clip(img, 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------
# Defect injection — each returns (modified_image, binary_mask)
# --------------------------------------------------------------------------
def _defect_region(size: int, rng: np.random.Generator) -> tuple[float, float]:
    """Pick a point on the part body (not the background, not the bore)."""
    for _ in range(200):
        ang = rng.uniform(0, 2 * np.pi)
        r = size * rng.uniform(0.12, 0.34)
        x, y = size / 2 + r * np.cos(ang), size / 2 + r * np.sin(ang)
        if 8 < x < size - 8 and 8 < y < size - 8:
            return float(x), float(y)
    return size / 2, size / 2


def _apply_scratch(img, rng):
    """A thin bright line: material displaced by contact."""
    size = img.shape[0]
    mask = np.zeros_like(img)
    x0, y0 = _defect_region(size, rng)
    ang = rng.uniform(0, np.pi)
    length = size * rng.uniform(0.10, 0.26)
    width = rng.uniform(0.7, 1.8)
    steps = int(length * 3)
    t = np.linspace(-length / 2, length / 2, steps)
    # Small random walk so the scratch is not a perfect straight line.
    wobble = gaussian_filter(rng.standard_normal(steps), 6.0) * 2.5
    xs = x0 + t * np.cos(ang) - wobble * np.sin(ang)
    ys = y0 + t * np.sin(ang) + wobble * np.cos(ang)
    for x, y in zip(xs, ys):
        xi, yi = int(round(x)), int(round(y))
        if 1 <= xi < size - 1 and 1 <= yi < size - 1:
            mask[yi, xi] = 1.0
    mask = gaussian_filter(mask, width)
    mask = mask / (mask.max() + 1e-8)
    out = img + rng.uniform(0.20, 0.38) * mask
    return np.clip(out, 0, 1), (mask > 0.25).astype(np.uint8)


def _apply_dent(img, rng):
    """A soft dark depression: an indentation that shadows."""
    size = img.shape[0]
    x0, y0 = _defect_region(size, rng)
    radius = size * rng.uniform(0.025, 0.055)
    yy, xx = np.mgrid[0:size, 0:size]
    d = np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2)
    blob = np.clip(1 - (d / radius) ** 2, 0, 1) ** 1.5
    blob = gaussian_filter(blob, 1.5)
    out = img - rng.uniform(0.16, 0.30) * blob
    # Highlight on one side sells the 3D depression look.
    shift = int(radius * 0.5)
    out += 0.07 * np.roll(np.roll(blob, -shift, 0), -shift, 1)
    return np.clip(out, 0, 1), (blob > 0.18).astype(np.uint8)


def _apply_contamination(img, rng):
    """Scattered dark specks: dust, swarf, oil spatter."""
    size = img.shape[0]
    x0, y0 = _defect_region(size, rng)
    mask = np.zeros_like(img)
    for _ in range(rng.integers(6, 18)):
        px = int(np.clip(x0 + rng.normal(0, size * 0.030), 2, size - 3))
        py = int(np.clip(y0 + rng.normal(0, size * 0.030), 2, size - 3))
        rad = int(rng.integers(1, 4))
        mask[py - rad : py + rad + 1, px - rad : px + rad + 1] = 1.0
    mask = gaussian_filter(mask, 1.0)
    mask = mask / (mask.max() + 1e-8)
    out = img - rng.uniform(0.25, 0.42) * mask
    return np.clip(out, 0, 1), (mask > 0.30).astype(np.uint8)


def _apply_crack(img, rng):
    """A dark branching fracture: the hardest class, thin and low-contrast."""
    size = img.shape[0]
    mask = np.zeros_like(img)
    x, y = _defect_region(size, rng)
    ang = rng.uniform(0, 2 * np.pi)
    for _ in range(int(size * rng.uniform(0.14, 0.30))):
        ang += rng.normal(0, 0.30)  # jagged, unlike a scratch
        x += np.cos(ang)
        y += np.sin(ang)
        xi, yi = int(round(x)), int(round(y))
        if 1 <= xi < size - 1 and 1 <= yi < size - 1:
            mask[yi, xi] = 1.0
        else:
            break
    mask = gaussian_filter(mask, 0.9)
    mask = mask / (mask.max() + 1e-8)
    out = img - rng.uniform(0.18, 0.30) * mask
    return np.clip(out, 0, 1), (mask > 0.30).astype(np.uint8)


_DEFECT_FNS = {
    "scratch": _apply_scratch,
    "dent": _apply_dent,
    "contamination": _apply_contamination,
    "crack": _apply_crack,
}


# --------------------------------------------------------------------------
# Dataset writer
# --------------------------------------------------------------------------
def _to_pil(arr: np.ndarray) -> Image.Image:
    """Save as 3-channel RGB even though the source is grey.

    Reason: ImageNet backbones expect 3 channels. Doing the conversion once, at
    write time, means every downstream loader is simpler.
    """
    u8 = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(np.stack([u8] * 3, axis=-1), mode="RGB")


def generate_dataset(
    root: str | Path,
    n_train_good: int = 220,
    n_test_good: int = 60,
    n_per_defect: int = 25,
    image_size: int = 256,
    seed: int = 7,
) -> dict[str, int]:
    """Write a full MVTec-style dataset to `root`. Returns per-split counts."""
    root = Path(root)
    rng = np.random.default_rng(seed)
    counts: dict[str, int] = {}

    train_dir = root / "train" / "good"
    train_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_train_good):
        _to_pil(_render_good_part(image_size, rng)).save(train_dir / f"{i:03d}.png")
    counts["train/good"] = n_train_good

    test_good = root / "test" / "good"
    test_good.mkdir(parents=True, exist_ok=True)
    for i in range(n_test_good):
        _to_pil(_render_good_part(image_size, rng)).save(test_good / f"{i:03d}.png")
    counts["test/good"] = n_test_good

    for dtype in DEFECT_TYPES:
        img_dir = root / "test" / dtype
        msk_dir = root / "ground_truth" / dtype
        img_dir.mkdir(parents=True, exist_ok=True)
        msk_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n_per_defect):
            base = _render_good_part(image_size, rng)
            defected, mask = _DEFECT_FNS[dtype](base, rng)
            _to_pil(defected).save(img_dir / f"{i:03d}.png")
            # MVTec stores masks as single-channel PNGs with 255 = defect.
            Image.fromarray((mask * 255).astype(np.uint8), mode="L").save(
                msk_dir / f"{i:03d}_mask.png"
            )
        counts[f"test/{dtype}"] = n_per_defect

    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a synthetic VisionQC dataset.")
    ap.add_argument("--root", default="data/synthetic")
    ap.add_argument("--train-good", type=int, default=220)
    ap.add_argument("--test-good", type=int, default=60)
    ap.add_argument("--per-defect", type=int, default=25)
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    counts = generate_dataset(
        args.root, args.train_good, args.test_good,
        args.per_defect, args.image_size, args.seed,
    )
    total = sum(counts.values())
    print(f"Wrote {total} images to {args.root}")
    for k, v in sorted(counts.items()):
        print(f"  {k:26s} {v:5d}")


if __name__ == "__main__":
    main()
