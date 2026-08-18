"""API request/response schemas.

WHY DEFINE SCHEMAS AT ALL
-------------------------
You could return a bare dict and FastAPI would happily serialise it. Defining
Pydantic models instead buys three things for almost no effort:

1. **A contract.** The client knows exactly which fields exist and their types.
   Change one and the schema changes visibly, instead of silently breaking a
   consumer at runtime.
2. **Free interactive docs.** FastAPI generates an OpenAPI spec from these
   classes, so `/docs` gives you a live page where anyone can upload an image
   and see the response. That page *is* your demo — it takes zero extra work and
   looks far more like a real product than a notebook cell.
3. **Validation at the boundary.** Bad data is rejected at the edge with a clear
   422 rather than causing a confusing exception three layers deep.

These are Pydantic v2 models (`Field(...)`, `model_config`), which is what
current FastAPI expects. Pydantic v1 syntax such as an inner `class Config` is
deprecated and will bite you.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"] = Field(
        description="'degraded' means the service is up but a model is missing."
    )
    version: str
    device: str
    models_loaded: list[str]
    classes: list[str]
    thresholds_calibrated: bool
    image_size: int


class DecisionPayload(BaseModel):
    verdict: Literal["PASS", "FAIL_CLASSIFIED", "FAIL_ANOMALY", "REVIEW"]
    defect_probability: float | None = Field(
        None, description="P(defect) from the classifier, in [0, 1]."
    )
    predicted_class: str | None = Field(
        None, description="Most likely defect type, or 'good'."
    )
    class_confidence: float | None = None
    anomaly_score: float | None = Field(
        None, description="Raw Mahalanobis distance. Unbounded; higher = more unusual."
    )
    anomaly_normalised: float | None = Field(
        None,
        description=(
            "Anomaly score divided by the 95th percentile of good validation "
            "parts. 1.0 means 'as unusual as the most unusual normal parts'. "
            "NOT a probability -- do not present it as one."
        ),
    )
    classifier_flag: bool
    anomaly_flag: bool
    reasons: list[str] = Field(
        description="Human-readable trace of how the verdict was reached."
    )


class InspectResponse(BaseModel):
    decision: DecisionPayload
    class_probabilities: dict[str, float] | None = None
    latency_ms: float
    heatmap_png_base64: str | None = Field(
        None,
        description=(
            "Base64 PNG: original | anomaly heatmap | Grad-CAM. Omitted when "
            "include_heatmap=false, which roughly halves the response size."
        ),
    )
    model_config = {
        "json_schema_extra": {
            "example": {
                "decision": {
                    "verdict": "FAIL_ANOMALY",
                    "defect_probability": 0.41,
                    "predicted_class": "good",
                    "class_confidence": 0.59,
                    "anomaly_score": 27.4,
                    "anomaly_normalised": 2.2,
                    "classifier_flag": False,
                    "anomaly_flag": True,
                    "reasons": [
                        "classifier P(defect)=0.410 < threshold 0.717",
                        "anomaly score=27.400 >= threshold 11.461",
                        "deviation from normal with no matching known defect class",
                    ],
                },
                "latency_ms": 118.3,
            }
        }
    }


class ErrorResponse(BaseModel):
    detail: str
