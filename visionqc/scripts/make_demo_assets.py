"""Generate the figures that go in your README.

RUN:
    PYTHONPATH=src python scripts/make_demo_assets.py --config configs/synthetic.yaml

WHY THIS MATTERS MORE THAN YOU THINK
------------------------------------
A recruiter spends about thirty seconds on your GitHub page. They will not clone
your repo, install PyTorch, and run your evaluation. What they will do is scroll.

A README with a picture showing a defect and a heatmap landing on it
communicates more in two seconds than three paragraphs of metrics. This script
makes that picture reproducibly, so it always matches your current model rather
than being a screenshot from three weeks ago.

Produces:
    docs/demo_grid.png       one row per sample: original | anomaly | Grad-CAM
    docs/verdicts.png        a strip of PASS / FAIL results with scores
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from visionqc.config import Config
from visionqc.explain.overlay import denormalise, overlay_heatmap, save_image
from visionqc.inference import InspectionEngine
from visionqc.utils import get_device, get_logger, load_json

logger = get_logger("visionqc.demo")


def _label_strip(width: int, text: str, height: int = 26) -> np.ndarray:
    """A small caption bar. Uses the default PIL font so it needs no font file
    on the system -- one less thing to break inside a container."""
    img = Image.new("RGB", (width, height), (245, 245, 245))
    ImageDraw.Draw(img).text((6, 6), text, fill=(20, 20, 20))
    return np.array(img)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="docs")
    ap.add_argument("--per-class", type=int, default=1)
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    manifest = load_json(cfg.run_dir / "splits.json")
    engine = InspectionEngine(cfg.run_dir, get_device(cfg.device))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pick a few representative test images: one good, one of each defect type.
    by_label: dict[str, list[dict]] = {}
    for r in manifest["splits"]["test"]:
        by_label.setdefault(r["label"], []).append(r)

    rows: list[np.ndarray] = []
    strip: list[np.ndarray] = []

    for label in sorted(by_label, key=lambda k: (k != "good", k)):
        for rec in by_label[label][: args.per_class]:
            img = Image.open(rec["path"])
            decision, result = engine.inspect(img)
            panel = engine.render_panel(result)

            caption = (
                f"true: {label}   ->   {decision.verdict.value}"
                f"   anomaly={decision.anomaly_score:.1f}"
                if decision.anomaly_score is not None
                else f"true: {label}   ->   {decision.verdict.value}"
            )
            rows.append(np.concatenate(
                [_label_strip(panel.shape[1], caption), panel], axis=0
            ))

            base = denormalise(result.image_tensor, engine.mean, engine.std)
            tile = base if result.anomaly_map is None else overlay_heatmap(
                base, result.anomaly_map
            )
            strip.append(np.concatenate(
                [_label_strip(tile.shape[1], f"{label} -> {decision.verdict.value}", 22),
                 tile], axis=0
            ))
            logger.info("%-14s -> %-16s anomaly=%s", label, decision.verdict.value,
                        f"{decision.anomaly_score:.2f}" if decision.anomaly_score else "n/a")

    if rows:
        width = max(r.shape[1] for r in rows)
        padded = [
            np.pad(r, ((0, 0), (0, width - r.shape[1]), (0, 0)),
                   constant_values=255) if r.shape[1] < width else r
            for r in rows
        ]
        save_image(np.concatenate(padded, axis=0), out_dir / "demo_grid.png")
        logger.info("Wrote %s", out_dir / "demo_grid.png")

    if strip:
        save_image(np.concatenate(strip, axis=1), out_dir / "verdicts.png")
        logger.info("Wrote %s", out_dir / "verdicts.png")

    logger.info("Panel layout: original | anomaly heatmap | Grad-CAM "
                "(whichever models are loaded)")


if __name__ == "__main__":
    main()
