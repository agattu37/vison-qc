# Getting the datasets

All three options produce the same canonical layout, so every downstream script
works identically.

```
data/<dataset>/
├── train/good/                 NORMAL ONLY — fits the anomaly model
├── test/good/
├── test/<defect_type>/         one folder per defect type
└── ground_truth/<type>/*_mask.png    optional pixel masks (255 = defect)
```

## 1. Synthetic (no download — start here)

```bash
python -m visionqc.data.synthetic --root data/synthetic \
    --train-good 220 --test-good 60 --per-defect 25
```

380 images in about a minute. Brushed-metal discs with scratch, dent,
contamination and crack defects, plus exact ground-truth masks.

**Honest limitation:** synthetic defects are easier than real ones — the noise
model is known and the lighting is simpler than a real factory. Use this to
prove the pipeline works; report MVTec numbers as your actual result.

## 2. MVTec AD (your real benchmark)

The standard academic and industrial benchmark for this exact problem. Free for
research and educational use; requires a short form.

1. Search for "MVTec Anomaly Detection dataset download" and follow the form.
2. Extract the archive (~5 GB, 15 categories).
3. **Start with ONE category** — `bottle`, `hazelnut` or `screw` are small and
   visually clear.

```bash
python scripts/prepare_dataset.py mvtec \
    --src ~/Downloads/mvtec_anomaly_detection/bottle \
    --dst data/mvtec/bottle

make all CONFIG=configs/mvtec_bottle.yaml RUN_DIR=artifacts/mvtec_bottle
```

Add `--link` to hard-link instead of copying (saves ~5 GB; same filesystem only).

**Reference numbers:** the PaDiM paper reports roughly 0.89 mean image-level
AUROC across all 15 categories with ResNet18 + 100 random dimensions, and around
0.98 with a much larger WideResNet50 backbone. Easy categories like `bottle`
typically score well above the mean. **Verify against the paper before quoting
these in an interview.**

## 3. Kaggle casting product dataset

~7,300 labelled images of submersible pump impellers, defective vs OK. Search
Kaggle for "Casting Product Image Data for Quality Inspection".

```bash
python scripts/prepare_dataset.py casting \
    --src ~/Downloads/casting_data --dst data/casting

make all CONFIG=configs/casting.yaml RUN_DIR=artifacts/casting
```

**No pixel masks**, so pixel-level AUROC reports `n/a`. State that in your
results rather than hiding it.

Why use a second dataset at all? "I validated on two domains" is a meaningfully
stronger claim than "I got a good number on one" — it's evidence your approach
generalises rather than that you got lucky.

## Adding your own dataset

Write ~30 lines in `scripts/prepare_dataset.py` following the existing
functions. Nothing downstream needs to change — that's the point of normalising
on disk rather than writing a Dataset class per source.
