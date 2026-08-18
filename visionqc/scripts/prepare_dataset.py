"""Convert real datasets into the MVTec-style layout this project expects.

WHY AN ADAPTER INSTEAD OF A LOADER PER DATASET
-----------------------------------------------
The tempting design is one Dataset class per source: `MVTecDataset`,
`CastingDataset`, `MyFactoryDataset`. Every one of them re-implements globbing,
label mapping and mask lookup, and they drift apart.

The better design is to normalise *on disk*, once, into a single canonical
layout. Then there is exactly one loader, one split builder, one evaluation
path. Adding a new dataset means writing ~30 lines here, not touching the model
code at all.

This is the same instinct as a schema in a data pipeline, and it is worth
naming as such in an interview: "I normalised heterogeneous sources into one
canonical layout at ingest, so nothing downstream needed to know where the data
came from."

CANONICAL LAYOUT
    <root>/train/good/*                 normal only -> fits the anomaly model
    <root>/test/good/*                  normal test samples
    <root>/test/<defect_type>/*         defective test samples
    <root>/ground_truth/<type>/*_mask.* optional pixel masks

USAGE
    # MVTec AD (after manual download + extraction)
    python scripts/prepare_dataset.py mvtec \
        --src ~/Downloads/mvtec_anomaly_detection/bottle --dst data/mvtec/bottle

    # Kaggle casting product dataset
    python scripts/prepare_dataset.py casting \
        --src ~/Downloads/casting_data --dst data/casting
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def _copy(files: list[Path], dst: Path, link: bool) -> int:
    """Copy or hard-link files into `dst`, renaming to a stable index.

    Hard links (`--link`) cost no extra disk. MVTec is ~5 GB; duplicating it to
    reshape a folder tree is wasteful. Links are the default-off option because
    they surprise people on Windows and across filesystems -- copying is the
    safe default, linking is the informed choice.
    """
    dst.mkdir(parents=True, exist_ok=True)
    for i, src in enumerate(files):
        target = dst / f"{i:04d}{src.suffix.lower()}"
        if target.exists():
            continue
        if link:
            try:
                target.hardlink_to(src)
                continue
            except OSError:
                pass  # different filesystem -> fall back to copy
        shutil.copy2(src, target)
    return len(files)


def prepare_mvtec(src: Path, dst: Path, link: bool) -> dict[str, int]:
    """MVTec AD is already in the canonical layout, so this mostly validates.

    We still copy rather than symlink the whole tree, because the split builder
    writes nothing but reads a lot, and having a self-contained `data/` folder
    makes the project easier to hand to someone else.
    """
    if not (src / "train" / "good").is_dir():
        raise FileNotFoundError(
            f"{src} does not look like an MVTec category folder.\n"
            f"Expected {src}/train/good/. Point --src at a single category "
            f"(e.g. .../mvtec_anomaly_detection/bottle), not the whole archive."
        )
    counts: dict[str, int] = {}
    counts["train/good"] = _copy(_images(src / "train" / "good"), dst / "train" / "good", link)

    for sub in sorted(p for p in (src / "test").iterdir() if p.is_dir()):
        counts[f"test/{sub.name}"] = _copy(_images(sub), dst / "test" / sub.name, link)

    gt = src / "ground_truth"
    if gt.is_dir():
        for sub in sorted(p for p in gt.iterdir() if p.is_dir()):
            # MVTec masks are named 000_mask.png and must stay aligned with
            # 000.png in the matching test folder. We preserve the index by
            # copying in the same sorted order, then renaming to <i>_mask.png.
            files = _images(sub)
            out = dst / "ground_truth" / sub.name
            out.mkdir(parents=True, exist_ok=True)
            for i, f in enumerate(files):
                target = out / f"{i:04d}_mask{f.suffix.lower()}"
                if not target.exists():
                    shutil.copy2(f, target)
            counts[f"ground_truth/{sub.name}"] = len(files)
    return counts


def prepare_casting(src: Path, dst: Path, link: bool) -> dict[str, int]:
    """Kaggle 'Casting Product Image Data for Quality Inspection'.

    Source layout (the archive nests one level, so we search for it):
        <src>/train/{ok_front, def_front}
        <src>/test/{ok_front, def_front}

    Mapping decisions and why:
      - train/ok_front  -> train/good   : the normal pool the anomaly model fits
      - test/ok_front   -> test/good
      - both def_front  -> test/defect  : ALL labelled defects go into the
        labelled pool, and our split builder then holds out a stratified 50% of
        them for the test set. Putting some defects in `train/` instead would
        break the invariant that `train/` is normal-only, which the anomaly path
        depends on.

    This dataset has no pixel masks, so pixel-level AUROC will report `n/a`.
    That is expected and worth stating in your results rather than hiding.
    """
    base = src
    if not (base / "train").is_dir():
        # The Kaggle zip commonly extracts to casting_data/casting_data/...
        for cand in src.rglob("train"):
            if (cand / "ok_front").is_dir() or (cand / "def_front").is_dir():
                base = cand.parent
                break
    if not (base / "train").is_dir():
        raise FileNotFoundError(
            f"Could not find a train/ folder with ok_front/def_front under {src}."
        )

    def pick(split: str, name: str) -> list[Path]:
        for variant in (name, name.replace("_front", "")):
            p = base / split / variant
            if p.is_dir():
                return _images(p)
        return []

    counts: dict[str, int] = {}
    counts["train/good"] = _copy(pick("train", "ok_front"), dst / "train" / "good", link)
    counts["test/good"] = _copy(pick("test", "ok_front"), dst / "test" / "good", link)

    defects = pick("train", "def_front") + pick("test", "def_front")
    counts["test/defect"] = _copy(defects, dst / "test" / "defect", link)

    if not counts["train/good"] or not counts["test/defect"]:
        raise ValueError(f"Found too little data under {base}: {counts}")
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=["mvtec", "casting"])
    ap.add_argument("--src", required=True, help="extracted download location")
    ap.add_argument("--dst", required=True, help="destination in canonical layout")
    ap.add_argument("--link", action="store_true",
                    help="hard-link instead of copying (saves disk, same filesystem only)")
    args = ap.parse_args()

    src, dst = Path(args.src).expanduser(), Path(args.dst).expanduser()
    if not src.exists():
        raise SystemExit(f"Source not found: {src}")

    fn = prepare_mvtec if args.dataset == "mvtec" else prepare_casting
    counts = fn(src, dst, args.link)

    print(f"\nPrepared '{args.dataset}' -> {dst}")
    for k, v in sorted(counts.items()):
        print(f"  {k:30s} {v:6d}")
    print(f"\nNext:\n  python -m visionqc.data.splits --root {dst} "
          f"--out artifacts/<run>/splits.json")


if __name__ == "__main__":
    main()
