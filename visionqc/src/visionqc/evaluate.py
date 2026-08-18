"""Evaluate everything, calibrate thresholds, and produce the report.

RUN:
    python -m visionqc.evaluate --config configs/synthetic.yaml

WHAT THIS PRODUCES
------------------
    artifacts/<run>/thresholds.json     calibrated operating points
    artifacts/<run>/results.json        every metric, machine-readable
    artifacts/<run>/results.md          the table you paste into your README
    artifacts/<run>/plots/*.png         ROC, cost curve, score distributions
    artifacts/<run>/failures/*.png      the error-analysis panels
    artifacts/<run>/predictions.csv     per-image scores, for your own digging

THE TWO-STAGE PROTOCOL, AND WHY IT IS NON-NEGOTIABLE
-----------------------------------------------------
    Stage 1  CALIBRATE on the validation split  -> pick thresholds
    Stage 2  EVALUATE  on the test split        -> report numbers

You may look at validation results as often as you like. You look at test
results once, at the end, with the thresholds already frozen.

If you instead sweep thresholds on the test set and report the best one, your
numbers are optimistically biased -- you have fitted a parameter to your test
data. It is a small leak but a real one, and an interviewer who asks "how did
you pick your threshold?" is usually checking for exactly this.

ERROR ANALYSIS IS NOT OPTIONAL
------------------------------
The PRD calls it the most interview-impressive part of the project, and that is
right. Anyone can print an AUROC. Being able to say "here are my six false
negatives, five of them are the low-contrast crack class, and here is why that
class is hard for a reconstruction-based method" demonstrates that you looked at
your data. We dump the worst failures as image panels automatically so you have
no excuse not to look.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # no display in a container; must be set before pyplot
import matplotlib.pyplot as plt
import numpy as np
import torch

from .config import Config
from .data.datasets import InspectionDataset, make_loader
from .decision import Thresholds, decide
from .inference import InspectionEngine
from .metrics import (
    binary_metrics, cost_curve, localisation_hit_rate, multiclass_report,
    pixel_auroc, roc_points, safe_auroc, select_threshold_by_cost,
    select_threshold_by_recall,
)
from .utils import get_device, get_logger, load_json, save_json, set_seed

logger = get_logger("visionqc.evaluate")


def _save_fig(fig, path) -> None:
    """Save a figure, creating parent directories first.

    Every plot goes through here. Doing the mkdir inside each plotting function
    is how you end up with one that forgets and crashes an hour into a run.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------
def collect(engine: InspectionEngine, records, class_to_idx, cfg,
            want_maps: bool = False) -> dict[str, Any]:
    """Run the engine over a split and gather everything we might need."""
    d = cfg.data
    ds = InspectionDataset(records, class_to_idx, d.image_size, d.mean, d.std,
                           train=False, load_masks=want_maps)
    loader = make_loader(ds, cfg.classifier.batch_size, False, d.num_workers)

    out: dict[str, list] = {
        "paths": [], "y_binary": [], "y_class": [], "label_names": [],
        "clf_prob": [], "clf_pred": [], "anomaly": [],
        "masks": [], "amaps": [], "images": [],
    }

    for batch in loader:
        x = batch["image"].to(engine.device)
        results = engine.score_batch(x, want_maps=want_maps, want_gradcam=False)
        for i, r in enumerate(results):
            out["paths"].append(batch["path"][i])
            out["y_binary"].append(int(batch["is_defect"][i]))
            out["y_class"].append(int(batch["label"][i]))
            out["clf_prob"].append(r.defect_probability)
            out["clf_pred"].append(
                engine.classes.index(r.predicted_class) if r.predicted_class else None
            )
            out["anomaly"].append(r.anomaly_score)
            if want_maps:
                out["masks"].append(batch["mask"][i].numpy())
                out["amaps"].append(
                    r.anomaly_map.numpy() if r.anomaly_map is not None else None
                )
                out["images"].append(r.image_tensor)
    return out


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_score_distribution(y, scores, title, path, threshold=None):
    """Overlapping histograms of good vs defect scores.

    This is the most informative single plot in the project. AUROC compresses
    the whole picture into one number; this shows you *why* it is what it is --
    how much the two populations overlap, whether the defects form one cluster
    or several, and whether any threshold could separate them at all.
    """
    y, scores = np.asarray(y), np.asarray(scores, dtype=float)
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(np.nanmin(scores), np.nanmax(scores), 40)
    ax.hist(scores[y == 0], bins=bins, alpha=0.65, label="good", color="#2a9d8f")
    ax.hist(scores[y == 1], bins=bins, alpha=0.65, label="defect", color="#e76f51")
    if threshold is not None:
        ax.axvline(threshold, color="k", ls="--", lw=1.6,
                   label=f"threshold = {threshold:.3g}")
    ax.set_xlabel("score"); ax.set_ylabel("count"); ax.set_title(title)
    ax.legend()
    _save_fig(fig, path)


def plot_roc(curves: dict[str, dict], path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    for name, c in curves.items():
        if c["fpr"]:
            ax.plot(c["fpr"], c["tpr"], lw=2, label=f"{name} (AUROC={c['auroc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="random")
    ax.set_xlabel("False positive rate (good parts wrongly flagged)")
    ax.set_ylabel("True positive rate (defects caught)")
    ax.set_title("ROC -- image level"); ax.legend(loc="lower right")
    _save_fig(fig, path)


def plot_cost_curve(curve: dict, chosen: float, cost_fn: float, cost_fp: float,
                    path: str | Path) -> None:
    """The plot that justifies your operating point to a non-ML stakeholder."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(curve["threshold"], curve["cost"], lw=2, color="#264653",
            label=f"cost = {cost_fn:g}*FN + {cost_fp:g}*FP")
    ax.axvline(chosen, color="#e76f51", ls="--", lw=1.8,
               label=f"chosen = {chosen:.3g}")
    ax.set_xlabel("threshold"); ax.set_ylabel("total cost (validation split)")
    ax.set_title("Cost-based threshold selection"); ax.legend()
    _save_fig(fig, path)


def plot_confusion(cm, classes, path: str | Path) -> None:
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(1.3 * len(classes) + 2.5,) * 2)
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title("Confusion matrix -- test split")
    thresh = cm.max() / 2 if cm.max() else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    _save_fig(fig, path)


def plot_history(path_json: Path, out: Path, keys: list[str], title: str) -> None:
    if not path_json.exists():
        return
    hist = load_json(path_json)
    if not hist:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    for k in keys:
        if k in hist[0]:
            ax.plot([h["epoch"] for h in hist], [h[k] for h in hist], marker="o",
                    ms=3, label=k)
    ax.set_xlabel("epoch"); ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)
    _save_fig(fig, out)


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------
def dump_failures(engine, data, threshold, out_dir: Path, classes, top_k: int = 6):
    """Save the worst false negatives and false positives as image panels.

    We rank by *how badly* the model got it wrong -- the most confidently missed
    defects and the most confidently flagged good parts -- because those are the
    informative cases. A borderline miss tells you your threshold is close; a
    confident miss tells you something is structurally wrong.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    y = np.array(data["y_binary"])
    s = np.array([v if v is not None else np.nan for v in data["anomaly"]], dtype=float)
    if np.isnan(s).all():
        return {"false_negatives": [], "false_positives": []}

    fn_idx = [i for i in np.where((y == 1) & (s < threshold))[0]]
    fp_idx = [i for i in np.where((y == 0) & (s >= threshold))[0]]
    fn_idx.sort(key=lambda i: s[i])            # lowest score = most confidently missed
    fp_idx.sort(key=lambda i: -s[i])           # highest score = most confident false alarm

    record = {"false_negatives": [], "false_positives": []}
    for tag, idxs in (("fn", fn_idx[:top_k]), ("fp", fp_idx[:top_k])):
        for rank, i in enumerate(idxs):
            from .inference import ScoreResult
            r = ScoreResult(
                image_tensor=data["images"][i],
                anomaly_map=torch.from_numpy(data["amaps"][i])
                if data["amaps"][i] is not None else None,
            )
            panel = engine.render_panel(r)
            name = f"{tag}_{rank:02d}_score{s[i]:.2f}_{Path(data['paths'][i]).parent.name}.png"
            from .explain.overlay import save_image
            save_image(panel, out_dir / name)
            key = "false_negatives" if tag == "fn" else "false_positives"
            record[key].append({
                "path": data["paths"][i],
                "true_class": classes[data["y_class"][i]],
                "anomaly_score": float(s[i]),
                "panel": str(out_dir / name),
            })
    logger.info("Error analysis: %d FN, %d FP panels -> %s",
                len(record["false_negatives"]), len(record["false_positives"]), out_dir)
    return record


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def write_markdown(results: dict, path: Path) -> None:
    L: list[str] = ["# VisionQC — Results", ""]
    L += [f"Run: `{results['run_name']}` · dataset: `{results['dataset']}` · "
          f"device: `{results['device']}`", ""]
    L += ["> Numbers below come from the held-out test split, using thresholds "
          "calibrated on validation only.", ""]

    L += ["## Image-level detection", "",
          "| Model | AUROC | AUPR | Recall | Precision | F1 | FN | FP |",
          "|---|---|---|---|---|---|---|---|"]
    for name, m in results["image_level"].items():
        L.append(
            f"| {name} | {m['auroc']:.4f} | {m['aupr']:.4f} | {m['recall']:.3f} | "
            f"{m['precision']:.3f} | {m['f1']:.3f} | {m['fn']} | {m['fp']} |"
        )
    L.append("")

    if results.get("localisation"):
        L += ["## Localisation (pixel level)", "",
              "| Model | Pixel AUROC | Peak-hit rate | Mean IoU (top 1%) |",
              "|---|---|---|---|"]
        for name, m in results["localisation"].items():
            L.append(
                f"| {name} | {_fmt(m.get('pixel_auroc'))} | "
                f"{_fmt(m.get('peak_hit_rate'))} | {_fmt(m.get('mean_iou'))} |"
            )
        L.append("")

    if results.get("classifier_multiclass"):
        mc = results["classifier_multiclass"]
        L += ["## Defect-type classification", "",
              f"Macro-recall **{mc['macro_recall']:.3f}** · "
              f"macro-F1 **{mc['macro_f1']:.3f}** · accuracy {mc['accuracy']:.3f}", "",
              "| Class | Precision | Recall | F1 | Support |", "|---|---|---|---|---|"]
        for c, v in mc["per_class"].items():
            L.append(f"| {c} | {v['precision']:.3f} | {v['recall']:.3f} | "
                     f"{v['f1']:.3f} | {v['support']} |")
        L.append("")

    L += ["## Operating point", ""]
    for k, v in results["thresholds"].items():
        if v is not None:
            L.append(f"- **{k}**: `{v:.6g}`")
    L.append("")
    for name, c in results.get("threshold_choices", {}).items():
        L.append(f"- _{name}_: {c['rationale']}")
    L.append("")

    if results.get("latency"):
        lat = results["latency"]
        L += ["## Latency (single image, batch size 1)", "",
              f"- mean **{lat['mean_ms']:.1f} ms** · p95 **{lat['p95_ms']:.1f} ms** "
              f"· device `{lat['device']}`",
              f"- PRD budget: < 2000 ms — **{'PASS' if lat['p95_ms'] < 2000 else 'FAIL'}**", ""]

    ea = results.get("error_analysis", {})
    if ea:
        L += ["## Error analysis", "",
              f"- False negatives inspected: {len(ea.get('false_negatives', []))}",
              f"- False positives inspected: {len(ea.get('false_positives', []))}", ""]
        if ea.get("false_negatives"):
            L += ["Most confidently missed defects:", ""]
            for f in ea["false_negatives"][:5]:
                L.append(f"- `{Path(f['path']).name}` (true: **{f['true_class']}**, "
                         f"score {f['anomaly_score']:.2f})")
            L.append("")
        L += ["_Panels are in `failures/`. Look at them and write down what you see — "
              "that paragraph is what interviewers remember._", ""]

    path.write_text("\n".join(L), encoding="utf-8")


def _fmt(v) -> str:
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.4f}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate VisionQC.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--splits", default=None)
    ap.add_argument("--target-recall", type=float, default=0.90)
    ap.add_argument("--latency-n", type=int, default=25)
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    run_dir = cfg.run_dir
    manifest = load_json(Path(args.splits) if args.splits else run_dir / "splits.json")
    classes, c2i = manifest["classes"], manifest["class_to_idx"]

    engine = InspectionEngine(run_dir, device, anomaly_backend=cfg.decision.primary
                              if cfg.decision.primary in {"padim", "autoencoder"} else "padim")

    results: dict[str, Any] = {
        "run_name": cfg.run_name, "dataset": manifest["root"], "device": str(device),
        "classes": classes, "image_level": {}, "localisation": {},
        "threshold_choices": {},
    }

    # ---------------- Stage 1: calibrate on VALIDATION --------------------
    logger.info("=" * 62)
    logger.info("STAGE 1 — calibrating thresholds on the VALIDATION split")
    logger.info("=" * 62)
    val = collect(engine, manifest["splits"]["sup_val"], c2i, cfg, want_maps=False)
    thresholds = Thresholds()

    if engine.has_anomaly and len(set(val["y_binary"])) > 1:
        ch = select_threshold_by_cost(
            val["y_binary"], val["anomaly"],
            cfg.decision.cost_false_negative, cfg.decision.cost_false_positive,
        )
        thresholds.anomaly = ch.threshold
        results["threshold_choices"]["anomaly (min cost)"] = ch.to_dict()
        logger.info("Anomaly threshold %.4f — %s", ch.threshold, ch.rationale)

        rch = select_threshold_by_recall(val["y_binary"], val["anomaly"], args.target_recall)
        results["threshold_choices"][f"anomaly (recall>={args.target_recall})"] = rch.to_dict()

        good_scores = [s for s, y in zip(val["anomaly"], val["y_binary"]) if y == 0]
        if good_scores:
            thresholds.anomaly_p95_good = float(np.percentile(good_scores, 95))

        curve = cost_curve(val["y_binary"], val["anomaly"],
                           cfg.decision.cost_false_negative, cfg.decision.cost_false_positive)
        plot_cost_curve(curve, ch.threshold, cfg.decision.cost_false_negative,
                        cfg.decision.cost_false_positive, run_dir / "plots" / "cost_curve.png")
        plot_score_distribution(val["y_binary"], val["anomaly"],
                                "Anomaly score — validation", 
                                run_dir / "plots" / "val_anomaly_scores.png", ch.threshold)
    elif engine.has_anomaly:
        logger.warning(
            "Validation split has only one class — cannot calibrate the anomaly "
            "threshold. Falling back to the 95th percentile of validation scores."
        )
        thresholds.anomaly = float(np.percentile(val["anomaly"], 95))

    if engine.has_classifier and len(set(val["y_binary"])) > 1:
        ch = select_threshold_by_cost(
            val["y_binary"], val["clf_prob"],
            cfg.decision.cost_false_negative, cfg.decision.cost_false_positive,
        )
        thresholds.classifier = ch.threshold
        results["threshold_choices"]["classifier (min cost)"] = ch.to_dict()
        logger.info("Classifier threshold %.4f — %s", ch.threshold, ch.rationale)
    elif engine.has_classifier:
        thresholds.classifier = 0.5

    save_json(thresholds.to_dict(), run_dir / "thresholds.json")
    engine.thresholds = thresholds
    results["thresholds"] = thresholds.to_dict()

    # ---------------- Stage 2: evaluate on TEST ---------------------------
    logger.info("=" * 62)
    logger.info("STAGE 2 — evaluating on the held-out TEST split (thresholds frozen)")
    logger.info("=" * 62)
    test = collect(engine, manifest["splits"]["test"], c2i, cfg, want_maps=True)
    y = test["y_binary"]
    roc_curves: dict[str, dict] = {}

    if engine.has_classifier:
        m = binary_metrics(y, test["clf_prob"], thresholds.classifier or 0.5)
        results["image_level"]["classifier"] = m.to_dict()
        logger.info("classifier  | %s", m.summary())
        rp = roc_points(y, test["clf_prob"]); rp["auroc"] = m.auroc
        roc_curves["classifier"] = rp
        results["classifier_multiclass"] = multiclass_report(
            test["y_class"], test["clf_pred"], classes
        )
        plot_confusion(results["classifier_multiclass"]["confusion_matrix"], classes,
                       run_dir / "plots" / "confusion_matrix.png")

    if engine.has_anomaly:
        m = binary_metrics(y, test["anomaly"], thresholds.anomaly or 0.0)
        results["image_level"][engine.anomaly_backend] = m.to_dict()
        logger.info("%-11s | %s", engine.anomaly_backend, m.summary())
        rp = roc_points(y, test["anomaly"]); rp["auroc"] = m.auroc
        roc_curves[engine.anomaly_backend] = rp
        plot_score_distribution(y, test["anomaly"], "Anomaly score — test",
                                run_dir / "plots" / "test_anomaly_scores.png",
                                thresholds.anomaly)

        masks = [m_ for m_ in test["masks"]]
        amaps = [a for a in test["amaps"] if a is not None]
        if masks and amaps and len(masks) == len(amaps):
            loc = localisation_hit_rate(masks, amaps)
            loc["pixel_auroc"] = pixel_auroc(np.array(masks), np.array(amaps))
            results["localisation"][engine.anomaly_backend] = loc
            logger.info("%-11s | pixel AUROC %.4f | peak-hit %.3f | mean IoU %.4f",
                        engine.anomaly_backend, loc["pixel_auroc"],
                        loc["peak_hit_rate"], loc["mean_iou"])

    # ---- fused decision --------------------------------------------------
    if engine.has_classifier and engine.has_anomaly:
        fused = []
        for i in range(len(y)):
            dec = decide(thresholds, test["clf_prob"][i], classes[test["clf_pred"][i]],
                         None, test["anomaly"][i], primary="fusion")
            fused.append(int(dec.verdict.value != "PASS"))
        fused = np.array(fused)
        yv = np.array(y)
        tp = int(((yv == 1) & (fused == 1)).sum()); fn = int(((yv == 1) & (fused == 0)).sum())
        fp = int(((yv == 0) & (fused == 1)).sum()); tn = int(((yv == 0) & (fused == 0)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        results["image_level"]["fusion (OR)"] = {
            # AUROC/AUPR are undefined for a rule that outputs a hard decision
            # rather than a score -- reporting a made-up number here would be
            # dishonest, so we mark it explicitly.
            "auroc": float("nan"), "aupr": float("nan"),
            "recall": rec, "precision": prec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "accuracy": (tp + tn) / len(yv), "specificity": tn / (tn + fp) if tn + fp else 0.0,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn, "n": len(yv), "threshold": float("nan"),
        }
        logger.info("fusion(OR)  | recall=%.3f precision=%.3f  [TP=%d FP=%d TN=%d FN=%d]",
                    rec, prec, tp, fp, tn, fn)

    if roc_curves:
        plot_roc(roc_curves, run_dir / "plots" / "roc.png")
    plot_history(run_dir / "classifier_history.json", run_dir / "plots" / "clf_history.png",
                 ["train_loss", "val_loss", "val_macro_recall"], "Classifier training")
    plot_history(run_dir / "autoencoder_history.json", run_dir / "plots" / "ae_history.png",
                 ["train_mse", "val_mse"], "Autoencoder training")

    # ---- latency ---------------------------------------------------------
    if len(test["paths"]) and engine.has_anomaly:
        import time
        from PIL import Image
        n = min(args.latency_n, len(test["paths"]))
        # Warm up first: the first forward pass pays lazy-init and cache costs
        # that would otherwise pollute the mean and make your latency look worse
        # than it is.
        engine.inspect(Image.open(test["paths"][0]))
        times = []
        for p in test["paths"][:n]:
            img = Image.open(p)
            t0 = time.perf_counter()
            engine.inspect(img)
            times.append((time.perf_counter() - t0) * 1000)
        results["latency"] = {
            "mean_ms": float(np.mean(times)), "p50_ms": float(np.percentile(times, 50)),
            "p95_ms": float(np.percentile(times, 95)), "n": n, "device": str(device),
        }
        logger.info("Latency: mean %.1f ms | p95 %.1f ms (n=%d, %s)",
                    results["latency"]["mean_ms"], results["latency"]["p95_ms"], n, device)

    # ---- error analysis + artifacts --------------------------------------
    if engine.has_anomaly:
        results["error_analysis"] = dump_failures(
            engine, test, thresholds.anomaly or 0.0, run_dir / "failures", classes
        )

    with open(run_dir / "predictions.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "true_class", "is_defect", "clf_p_defect",
                    "clf_pred_class", "anomaly_score"])
        for i in range(len(test["paths"])):
            w.writerow([
                test["paths"][i], classes[test["y_class"][i]], test["y_binary"][i],
                test["clf_prob"][i],
                classes[test["clf_pred"][i]] if test["clf_pred"][i] is not None else "",
                test["anomaly"][i],
            ])

    save_json(results, run_dir / "results.json")
    write_markdown(results, run_dir / "results.md")
    logger.info("Wrote results.json, results.md, plots/, failures/, predictions.csv")
    logger.info("Report -> %s", run_dir / "results.md")


if __name__ == "__main__":
    main()
