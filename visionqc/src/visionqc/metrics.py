"""Metrics, threshold selection, and the cost reasoning behind them.

WHY ACCURACY IS THE WRONG HEADLINE METRIC HERE
----------------------------------------------
Suppose a line runs at 2% defect rate. A model that stamps "PASS" on everything
is 98% accurate and has caught zero defects. Accuracy rewards it. So we never
lead with accuracy.

What we report instead, and why:

* **Recall on defects** (= sensitivity, = 1 - miss rate). The fraction of real
  defects we caught. This is the number the plant manager cares about, because a
  missed defect leaves the factory, reaches a customer, and can trigger a
  recall.
* **Precision.** Of the parts we flagged, how many were really defective. Low
  precision means operators waste time re-inspecting good parts, and within a
  week they start ignoring the system. A model nobody trusts has zero value
  regardless of its recall.
* **AUROC.** Threshold-free ranking quality: the probability that a randomly
  chosen defective image scores higher than a randomly chosen good one. This is
  the standard MVTec AD metric, which is exactly why we report it -- it lets you
  put your number next to published results instead of in a vacuum.
* **AUPR (average precision).** AUROC can look flattering when negatives vastly
  outnumber positives, because a large absolute number of false positives is
  still a small *rate*. AUPR does not have that blind spot. Reporting both is
  the honest choice under class imbalance.

THE THRESHOLD IS A BUSINESS DECISION, NOT A DEFAULT
---------------------------------------------------
`score > 0.5` is a convention with no meaning in QC. The real question is: what
does each kind of mistake cost?

    total_cost(t) = C_fn * (missed defects) + C_fp * (false alarms)

Set C_fn = 10 and C_fp = 1 -- "one escaped defect hurts as much as ten
unnecessary re-inspections" -- and then simply pick the threshold minimising
total cost. That single line of reasoning converts a model score into an
operating decision, and it is one of the most interview-valuable things in this
project. When asked "why 0.31?", the answer is not "it worked best", it is
"because at our cost ratio that is the minimum-cost operating point, and here is
the curve".

CRITICAL: thresholds are chosen on the **validation** split, never on test.
Choosing an operating point on your test set means your reported numbers are
optimistically biased. It is a subtle leak, and it is extremely common.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score, confusion_matrix, precision_recall_fscore_support,
    roc_auc_score, roc_curve,
)


# ---------------------------------------------------------------------------
# Image-level binary metrics
# ---------------------------------------------------------------------------
@dataclass
class BinaryMetrics:
    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    specificity: float
    auroc: float
    aupr: float
    tp: int
    fp: int
    tn: int
    fn: int
    n: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"n={self.n}  thr={self.threshold:.4f}  "
            f"recall={self.recall:.3f}  precision={self.precision:.3f}  "
            f"f1={self.f1:.3f}  AUROC={self.auroc:.4f}  AUPR={self.aupr:.4f}  "
            f"[TP={self.tp} FP={self.fp} TN={self.tn} FN={self.fn}]"
        )


def binary_metrics(
    y_true: Sequence[int], scores: Sequence[float], threshold: float
) -> BinaryMetrics:
    """Full binary report at one operating point. 1 = defect, 0 = good."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=np.float64)
    if y.shape != s.shape:
        raise ValueError(f"Shape mismatch: labels {y.shape} vs scores {s.shape}")
    pred = (s >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return BinaryMetrics(
        threshold=float(threshold),
        accuracy=float((tp + tn) / len(y)),
        precision=float(precision), recall=float(recall), f1=float(f1),
        specificity=float(specificity),
        auroc=safe_auroc(y, s), aupr=safe_aupr(y, s),
        tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn), n=int(len(y)),
    )


def safe_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """AUROC that returns NaN instead of raising when only one class is present.

    This happens more often than you would think -- e.g. a per-defect-type
    breakdown where a slice contains no good samples. Crashing the whole
    evaluation over one degenerate slice is worse than reporting NaN.
    """
    y = np.asarray(y_true)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, scores))


def safe_aupr(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y_true)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, scores))


# ---------------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------------
@dataclass
class ThresholdChoice:
    threshold: float
    strategy: str
    expected_cost: float
    recall: float
    precision: float
    f1: float
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _candidate_thresholds(scores: np.ndarray, n: int = 512) -> np.ndarray:
    """Candidate cut points spanning the observed score range.

    We use quantiles of the actual scores rather than a uniform grid, because
    anomaly scores are usually heavily skewed -- a uniform grid would spend most
    of its candidates in an empty region and miss the interesting part.
    """
    qs = np.linspace(0.0, 1.0, n)
    cand = np.unique(np.quantile(scores, qs))
    # Extend slightly past both ends so "flag nothing" and "flag everything"
    # are both reachable operating points.
    span = float(scores.max() - scores.min()) or 1.0
    return np.concatenate([[scores.min() - 0.01 * span], cand,
                           [scores.max() + 0.01 * span]])


def select_threshold_by_cost(
    y_true: Sequence[int],
    scores: Sequence[float],
    cost_fn: float = 10.0,
    cost_fp: float = 1.0,
) -> ThresholdChoice:
    """Pick the threshold minimising expected business cost."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=np.float64)
    if len(np.unique(y)) < 2:
        raise ValueError(
            "Threshold selection needs both classes in the validation split. "
            "Check your split: sup_val must contain good AND defective samples."
        )

    best: tuple[float, float] | None = None
    for t in _candidate_thresholds(s):
        pred = (s >= t).astype(int)
        fn = int(((y == 1) & (pred == 0)).sum())
        fp = int(((y == 0) & (pred == 1)).sum())
        cost = cost_fn * fn + cost_fp * fp
        # Strict `<` means ties resolve to the LOWER threshold, which is the
        # more conservative (higher-recall) choice. In QC that is the right
        # tie-break: when in doubt, flag it for human review.
        if best is None or cost < best[1]:
            best = (float(t), float(cost))

    thr, cost = best  # type: ignore[misc]
    m = binary_metrics(y, s, thr)
    return ThresholdChoice(
        threshold=thr, strategy="min_cost", expected_cost=cost,
        recall=m.recall, precision=m.precision, f1=m.f1,
        rationale=(
            f"Minimises {cost_fn:g}*FN + {cost_fp:g}*FP on the validation split. "
            f"At this point: recall={m.recall:.3f}, precision={m.precision:.3f}, "
            f"FN={m.fn}, FP={m.fp}, total cost={cost:g}."
        ),
    )


def select_threshold_by_recall(
    y_true: Sequence[int], scores: Sequence[float], target_recall: float = 0.95
) -> ThresholdChoice:
    """Highest threshold that still achieves `target_recall`.

    Use when recall is a hard contract ("we must catch 95% of defects") rather
    than something to be traded off. Taking the *highest* such threshold gives
    you the best precision available subject to that constraint -- there is no
    reason to flag more parts than the contract requires.
    """
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=np.float64)
    feasible: list[tuple[float, float]] = []
    for t in _candidate_thresholds(s):
        pred = (s >= t).astype(int)
        tp = int(((y == 1) & (pred == 1)).sum())
        fn = int(((y == 1) & (pred == 0)).sum())
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        if rec >= target_recall:
            feasible.append((float(t), rec))

    if not feasible:
        # Degrade gracefully rather than crash: flag everything, report honestly.
        thr = float(s.min() - 1e-6)
        m = binary_metrics(y, s, thr)
        return ThresholdChoice(
            threshold=thr, strategy="recall_target_unreachable",
            expected_cost=float("nan"), recall=m.recall, precision=m.precision,
            f1=m.f1,
            rationale=(
                f"Target recall {target_recall:.2f} is unreachable at any threshold "
                f"on this split; falling back to flagging everything. This means "
                f"the model's ranking is too weak for this contract -- improve the "
                f"model, do not tune the threshold."
            ),
        )

    thr = max(t for t, _ in feasible)
    m = binary_metrics(y, s, thr)
    return ThresholdChoice(
        threshold=thr, strategy=f"recall>={target_recall:.2f}",
        expected_cost=float("nan"), recall=m.recall, precision=m.precision, f1=m.f1,
        rationale=(
            f"Highest threshold achieving recall >= {target_recall:.2f} on "
            f"validation, which maximises precision subject to that constraint. "
            f"Achieved recall={m.recall:.3f}, precision={m.precision:.3f}."
        ),
    )


def cost_curve(
    y_true: Sequence[int], scores: Sequence[float],
    cost_fn: float = 10.0, cost_fp: float = 1.0, n: int = 256,
) -> dict[str, list[float]]:
    """Cost vs threshold, for the plot that justifies your operating point."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=np.float64)
    ts = _candidate_thresholds(s, n)
    out: dict[str, list[float]] = {"threshold": [], "cost": [], "recall": [],
                                   "precision": [], "fn": [], "fp": []}
    for t in ts:
        pred = (s >= t).astype(int)
        tp = int(((y == 1) & (pred == 1)).sum())
        fn = int(((y == 1) & (pred == 0)).sum())
        fp = int(((y == 0) & (pred == 1)).sum())
        out["threshold"].append(float(t))
        out["cost"].append(float(cost_fn * fn + cost_fp * fp))
        out["recall"].append(tp / (tp + fn) if (tp + fn) else 0.0)
        out["precision"].append(tp / (tp + fp) if (tp + fp) else 0.0)
        out["fn"].append(float(fn))
        out["fp"].append(float(fp))
    return out


# ---------------------------------------------------------------------------
# Multiclass
# ---------------------------------------------------------------------------
def multiclass_report(
    y_true: Sequence[int], y_pred: Sequence[int], classes: Sequence[str]
) -> dict[str, Any]:
    """Per-class precision/recall/F1 plus the confusion matrix.

    `zero_division=0` prevents a warning storm for classes with no predictions,
    which happens routinely with rare defect types.
    """
    y, p = np.asarray(y_true), np.asarray(y_pred)
    labels = list(range(len(classes)))
    pr, rc, f1, sup = precision_recall_fscore_support(
        y, p, labels=labels, zero_division=0
    )
    cm = confusion_matrix(y, p, labels=labels)
    return {
        "classes": list(classes),
        "per_class": {
            c: {"precision": float(pr[i]), "recall": float(rc[i]),
                "f1": float(f1[i]), "support": int(sup[i])}
            for i, c in enumerate(classes)
        },
        "macro_f1": float(f1.mean()),
        # Macro recall weights every class equally, so a rare defect type cannot
        # be ignored. Micro/weighted averages would let the abundant 'good'
        # class dominate and hide a class we never detect.
        "macro_recall": float(rc.mean()),
        "accuracy": float((y == p).mean()),
        "confusion_matrix": cm.tolist(),
    }


# ---------------------------------------------------------------------------
# Pixel-level localisation
# ---------------------------------------------------------------------------
def pixel_auroc(
    masks: np.ndarray, amaps: np.ndarray,
    max_pixels: int = 2_000_000, seed: int = 0,
) -> float:
    """AUROC over individual pixels: does the heatmap land on the defect?

    This is the metric that separates "my model is right" from "my model is
    right for the right reason". A model can have perfect image-level AUROC
    while its heatmap points at the background.

    We subsample pixels above `max_pixels` because a full test set is tens of
    millions of pixels and ranking them all is slow with negligible accuracy
    gain. The subsample is seeded, so the number is reproducible.
    """
    y = np.asarray(masks).reshape(-1).astype(np.uint8)
    s = np.asarray(amaps).reshape(-1).astype(np.float64)
    if y.shape != s.shape:
        raise ValueError(f"Mask/map shape mismatch: {y.shape} vs {s.shape}")
    if len(np.unique(y)) < 2:
        return float("nan")
    if len(y) > max_pixels:
        rng = np.random.default_rng(seed)
        # Keep every defect pixel (they are rare and precious), subsample the
        # normal ones. Stratifying this way keeps the estimate stable.
        pos = np.flatnonzero(y == 1)
        neg = np.flatnonzero(y == 0)
        n_neg = max(1, max_pixels - len(pos))
        if len(neg) > n_neg:
            neg = rng.choice(neg, size=n_neg, replace=False)
        keep = np.concatenate([pos, neg])
        y, s = y[keep], s[keep]
    return float(roc_auc_score(y, s))


def localisation_iou(
    mask: np.ndarray, amap: np.ndarray, quantile: float = 0.99
) -> float:
    """IoU between the hottest region of the map and the ground-truth mask.

    We take the **top k pixels by score**, where k = (1 - quantile) * n, rather
    than thresholding at the quantile *value*. The two sound equivalent but are
    not when the map has ties or is degenerate: `amap >= np.quantile(amap, 0.99)`
    selects every pixel of a map that is 95% zeros, because the 99th percentile
    of that map is itself zero. Top-k always selects exactly k pixels and cannot
    blow up that way.

    We rank rather than use a fixed threshold because anomaly-score magnitudes
    are not comparable across images. Taking the top 1% asks a fair question:
    "of the region this model considers most suspicious, how much overlaps the
    real defect?"
    """
    m = np.asarray(mask).astype(bool)
    if not m.any():
        return float("nan")   # undefined for good images -- no region to hit
    a = np.asarray(amap, dtype=np.float64).reshape(-1)
    k = max(1, int(round((1.0 - quantile) * a.size)))
    # argpartition is O(n) vs O(n log n) for a full sort; on a 256x256 map
    # evaluated over hundreds of images that difference is worth having.
    top = np.argpartition(-a, k - 1)[:k]
    pred = np.zeros(a.size, dtype=bool)
    pred[top] = True
    pred = pred.reshape(m.shape)
    union = (pred | m).sum()
    return float((pred & m).sum() / union) if union else 0.0


def localisation_hit_rate(
    masks: list[np.ndarray], amaps: list[np.ndarray], top_frac: float = 0.01
) -> dict[str, float]:
    """Does the single hottest pixel fall inside the true defect region?

    A blunt but very readable metric, and it maps directly to the PRD's
    "heatmap overlaps the ground-truth region" success criterion. Reported
    alongside mean IoU because IoU alone punishes a correct-but-diffuse map
    harshly, and for an operator a correct-but-diffuse pointer is still useful.
    """
    hits, ious = 0, []
    n = 0
    for m, a in zip(masks, amaps):
        m = np.asarray(m).astype(bool)
        if not m.any():
            continue
        n += 1
        a = np.asarray(a)
        peak = np.unravel_index(np.argmax(a), a.shape)
        hits += bool(m[peak])
        ious.append(localisation_iou(m, a, 1.0 - top_frac))
    return {
        "n_defective": n,
        "peak_hit_rate": hits / n if n else float("nan"),
        "mean_iou": float(np.nanmean(ious)) if ious else float("nan"),
    }


def roc_points(y_true: Sequence[int], scores: Sequence[float]) -> dict[str, list[float]]:
    """FPR/TPR arrays for plotting the ROC curve."""
    y = np.asarray(y_true).astype(int)
    if len(np.unique(y)) < 2:
        return {"fpr": [], "tpr": [], "thresholds": []}
    fpr, tpr, thr = roc_curve(y, np.asarray(scores, dtype=np.float64))
    return {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": thr.tolist()}
