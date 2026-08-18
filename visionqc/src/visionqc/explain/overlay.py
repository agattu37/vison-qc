"""Turn an anomaly/attention map into something a human can look at.

WHY A SEPARATE MODULE
---------------------
Both explanation paths -- Grad-CAM (supervised) and PaDiM/autoencoder
(unsupervised) -- produce the same shape: a (H, W) float map. So the rendering
code should be written once and shared. Keeping it out of the model files also
means the models have zero dependency on matplotlib or PIL, which keeps them
importable in a minimal serving environment.

A NOTE ON COLOUR MAPS
---------------------
We default to `inferno`, not `jet`. `jet` is the classic heatmap look and it is
genuinely bad: it is not perceptually uniform, so it invents visual boundaries
where the data is smooth, and it is unreadable for the ~8% of men with red-green
colour blindness. `inferno` and `viridis` are perceptually uniform and
colour-blind safe. This is a small thing that signals you have thought about who
looks at your output.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import torch
from matplotlib import colormaps
from PIL import Image


def denormalise(
    img: torch.Tensor,
    mean: tuple[float, ...] = (0.485, 0.456, 0.406),
    std: tuple[float, ...] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """(3,H,W) normalised tensor -> (H,W,3) uint8 array.

    Every visualisation needs this. Forgetting it gives you the classic
    washed-out, over-saturated "why does my image look radioactive" plot.
    """
    if img.dim() == 4:
        img = img[0]
    m = torch.tensor(mean, device=img.device).view(3, 1, 1)
    s = torch.tensor(std, device=img.device).view(3, 1, 1)
    x = (img.detach() * s + m).clamp(0, 1)
    return (x.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


def normalise_map(
    amap: torch.Tensor | np.ndarray,
    vmin: float | None = None,
    vmax: float | None = None,
) -> np.ndarray:
    """Scale a map to [0,1] as float32.

    Pass explicit vmin/vmax when you want several images on a *shared* scale --
    for example when comparing a good part against a defective one. Otherwise
    each image self-normalises and a perfectly clean part will still show a
    vivid "hottest" region, which misleads viewers badly.
    """
    a = amap.detach().cpu().numpy() if isinstance(amap, torch.Tensor) else np.asarray(amap)
    a = a.astype(np.float32)
    lo = float(a.min()) if vmin is None else float(vmin)
    hi = float(a.max()) if vmax is None else float(vmax)
    if hi - lo < 1e-12:
        return np.zeros_like(a)
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0)


def colourise(amap01: np.ndarray, cmap: str = "inferno") -> np.ndarray:
    """[0,1] map -> (H,W,3) uint8 RGB using a perceptually uniform colour map."""
    rgba = colormaps[cmap](amap01)          # (H, W, 4) float in [0,1]
    return (rgba[..., :3] * 255).astype(np.uint8)


def overlay_heatmap(
    image_u8: np.ndarray,
    amap: torch.Tensor | np.ndarray,
    alpha: float = 0.45,
    cmap: str = "inferno",
    vmin: float | None = None,
    vmax: float | None = None,
    mask_below: float | None = 0.35,
) -> np.ndarray:
    """Blend a heatmap over the original image.

    `mask_below` makes cold regions fully transparent instead of tinting the
    whole image dark purple. Without it, a clean part is uniformly washed in
    colour and the operator cannot see the part at all. With it, only the
    regions that actually carry evidence are painted -- which is what a heatmap
    is *for*.
    """
    a01 = normalise_map(amap, vmin, vmax)
    if a01.shape != image_u8.shape[:2]:
        a01 = np.asarray(
            Image.fromarray((a01 * 255).astype(np.uint8)).resize(
                (image_u8.shape[1], image_u8.shape[0]), Image.BILINEAR
            )
        ).astype(np.float32) / 255.0

    heat = colourise(a01, cmap).astype(np.float32)
    base = image_u8.astype(np.float32)

    # Per-pixel alpha: 0 where the map is cold, `alpha` where it is hot.
    if mask_below is not None:
        strength = np.clip((a01 - mask_below) / max(1e-6, 1.0 - mask_below), 0, 1)
    else:
        strength = np.ones_like(a01)
    a = (strength * alpha)[..., None]

    return np.clip(base * (1 - a) + heat * a, 0, 255).astype(np.uint8)


def side_by_side(panels: list[np.ndarray], gap: int = 6, bg: int = 255) -> np.ndarray:
    """Concatenate equal-height panels horizontally with a separator."""
    if not panels:
        raise ValueError("side_by_side received no panels")
    h = max(p.shape[0] for p in panels)
    out: list[np.ndarray] = []
    for i, p in enumerate(panels):
        if p.ndim == 2:
            p = np.stack([p] * 3, axis=-1)
        if p.shape[0] != h:
            p = np.asarray(Image.fromarray(p).resize((p.shape[1], h), Image.BILINEAR))
        if i:
            out.append(np.full((h, gap, 3), bg, dtype=np.uint8))
        out.append(p)
    return np.concatenate(out, axis=1)


def to_png_bytes(arr: np.ndarray) -> bytes:
    """Encode to PNG in memory -- used by the API to return a heatmap inline."""
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def save_image(arr: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)
