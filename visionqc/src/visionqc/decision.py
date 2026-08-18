"""The decision layer: two model scores in, one actionable verdict out.

WHY THIS DESERVES ITS OWN MODULE
--------------------------------
Everything upstream produces numbers. A factory does not act on numbers, it acts
on decisions: does this part continue down the line, or get pulled? Putting that
translation in one small, tested, readable file -- rather than scattering
`if score > 0.5` through the API -- is the difference between a demo and a
system.

It is also the piece interviewers probe hardest, because it is where ML meets
the business.

THE FUSION RULE, AND WHY IT IS AN *OR*
--------------------------------------
The two paths fail in different, largely uncorrelated ways:

* The **classifier** knows the defect types it was trained on, and knows them
  well. It is blind to anything else. Show it a defect type that did not exist
  when you collected data -- a new supplier's contamination, a novel tool-wear
  pattern -- and it will confidently say "good", because "good" is the closest
  thing in its vocabulary.
* **PaDiM** never learned defect types at all. It only knows what normal looks
  like, so *any* deviation registers, including types nobody has ever labelled.
  In exchange, it cannot tell you which defect it is, and it is more easily
  upset by benign variation like a lighting change.

Because their blind spots barely overlap, flagging when *either* fires catches
strictly more defects than either alone. The cost is precision: false alarms
from both models add up.

That trade is the right one **in this domain**, and you should be able to say
why: a missed defect ships to a customer, while a false alarm costs one operator
30 seconds at a re-inspection station. When those costs are asymmetric by ~10x,
you buy recall with precision. In a domain where they are not -- say, flagging
transactions for a fraud team with limited capacity -- an AND rule or a learned
combiner would be the better choice. The rule follows from the cost structure,
not from ML fashion.

WHY NOT TRAIN A LEARNED FUSION MODEL
------------------------------------
Stacking a small logistic regression on [classifier_score, anomaly_score] is the
obvious next step and would probably help a little. We do not, deliberately: it
needs a third held-out split to fit honestly, and with tens of labelled defects
that split would be too small to trust. Recognising when you do not have the
data to justify a more complex method is a real engineering skill. It is listed
in Future Work, which is the correct place for it.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from enum import Enum
from typing import Any


class Verdict(str, Enum):
    """Inheriting from `str` makes this JSON-serialisable with no custom encoder,
    which matters because it crosses the API boundary."""

    PASS = "PASS"
    FAIL_CLASSIFIED = "FAIL_CLASSIFIED"      # defect found AND named
    FAIL_ANOMALY = "FAIL_ANOMALY"            # deviation found, type unknown
    REVIEW = "REVIEW"                        # models disagree -> human decides


@dataclass
class Decision:
    verdict: Verdict
    defect_probability: float | None
    predicted_class: str | None
    class_confidence: float | None
    anomaly_score: float | None
    anomaly_normalised: float | None
    classifier_flag: bool
    anomaly_flag: bool
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["verdict"] = self.verdict.value
        return d


@dataclass
class Thresholds:
    """Calibrated operating points. Both come from the validation split.

    `anomaly_p95_good` is the 95th percentile of anomaly scores on *good*
    validation images. We keep it purely to turn a raw, unbounded Mahalanobis
    distance into a human-readable 0-1 number for the UI. It never affects the
    verdict -- the raw threshold does. Mixing display scaling into decision
    logic is a good way to ship a bug you cannot reproduce.
    """

    classifier: float | None = None
    anomaly: float | None = None
    anomaly_p95_good: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Thresholds":
        return cls(
            classifier=d.get("classifier"),
            anomaly=d.get("anomaly"),
            anomaly_p95_good=d.get("anomaly_p95_good"),
        )


def normalise_anomaly(score: float, p95_good: float | None) -> float | None:
    """Map a raw distance onto a friendlier 0-1-ish scale for display.

    1.0 means "as anomalous as the 95th-percentile good part". Values above 1
    are more anomalous than almost any good part. This is a presentation
    convenience, explicitly not a probability -- do not let a stakeholder read
    0.8 as "80% chance of a defect", because it is not.
    """
    if p95_good is None or p95_good <= 0:
        return None
    return float(score / p95_good)


def decide(
    thresholds: Thresholds,
    defect_probability: float | None = None,
    predicted_class: str | None = None,
    class_confidence: float | None = None,
    anomaly_score: float | None = None,
    primary: str = "fusion",
) -> Decision:
    """Combine available signals into a verdict.

    Every argument is optional so the same function serves all deployment
    modes: classifier-only (you have labels), anomaly-only (you do not), or
    both. Missing signals simply do not vote -- they never silently default to
    "good", which would turn a model-loading failure into a silent stream of
    PASS verdicts. Failing loudly beats passing everything.
    """
    reasons: list[str] = []

    clf_flag = False
    if defect_probability is not None and thresholds.classifier is not None:
        clf_flag = defect_probability >= thresholds.classifier
        reasons.append(
            f"classifier P(defect)={defect_probability:.3f} "
            f"{'>=' if clf_flag else '<'} threshold {thresholds.classifier:.3f}"
        )

    ano_flag = False
    if anomaly_score is not None and thresholds.anomaly is not None:
        ano_flag = anomaly_score >= thresholds.anomaly
        reasons.append(
            f"anomaly score={anomaly_score:.3f} "
            f"{'>=' if ano_flag else '<'} threshold {thresholds.anomaly:.3f}"
        )

    if not reasons:
        reasons.append("No calibrated signal available; defaulting to REVIEW.")
        return Decision(
            verdict=Verdict.REVIEW, defect_probability=defect_probability,
            predicted_class=predicted_class, class_confidence=class_confidence,
            anomaly_score=anomaly_score,
            anomaly_normalised=normalise_anomaly(
                anomaly_score, thresholds.anomaly_p95_good
            ) if anomaly_score is not None else None,
            classifier_flag=False, anomaly_flag=False, reasons=reasons,
        )

    # --- routing -----------------------------------------------------------
    if primary == "classifier":
        flagged = clf_flag
    elif primary in {"padim", "autoencoder", "anomaly"}:
        flagged = ano_flag
    elif primary == "fusion":
        flagged = clf_flag or ano_flag
    else:
        raise ValueError(
            f"Unknown decision.primary='{primary}'. Use one of: "
            "classifier, padim, autoencoder, anomaly, fusion."
        )

    if not flagged:
        verdict = Verdict.PASS
    elif clf_flag and predicted_class and predicted_class != "good":
        # The classifier recognised the defect, so we can name it. Most useful
        # verdict: the operator knows what to look for and which bin it goes in.
        verdict = Verdict.FAIL_CLASSIFIED
        reasons.append(f"classified as '{predicted_class}'")
    elif ano_flag:
        # The anomaly path fired and the classifier could not put a name to it
        # (either it did not fire, or it fired but its top class is still
        # 'good'). Either way there is real evidence of deviation without a
        # known defect type -- possibly a defect type nobody has labelled yet.
        # Routed for human review and labelling, which is exactly where an
        # active-learning loop would hook in later.
        verdict = Verdict.FAIL_ANOMALY
        reasons.append("deviation from normal with no matching known defect class")
    else:
        # Only the classifier fired, and its own top class is 'good' -- it is
        # internally inconsistent. Do not guess; send it to a human.
        verdict = Verdict.REVIEW
        reasons.append("classifier flagged but could not name a defect class")

    return Decision(
        verdict=verdict,
        defect_probability=defect_probability,
        predicted_class=predicted_class,
        class_confidence=class_confidence,
        anomaly_score=anomaly_score,
        anomaly_normalised=normalise_anomaly(
            anomaly_score, thresholds.anomaly_p95_good
        ) if anomaly_score is not None else None,
        classifier_flag=clf_flag,
        anomaly_flag=ano_flag,
        reasons=reasons,
    )
