"""FastAPI service exposing the inspection pipeline.

RUN LOCALLY:
    export VISIONQC_RUN_DIR=artifacts/synthetic
    uvicorn visionqc.api.main:app --reload --port 8000
    # then open http://localhost:8000/docs

ENDPOINTS
    GET  /health    liveness + what is actually loaded
    POST /inspect   image in, verdict + heatmap out
    GET  /          redirect to the interactive docs

THE ONE THING THAT MATTERS MOST HERE: LOAD MODELS ONCE
-------------------------------------------------------
Loading a checkpoint takes hundreds of milliseconds to seconds. Do it inside the
request handler and every request pays that cost, blowing the 2-second budget
immediately and pinning your CPU.

So we load at startup, into a module-level singleton, using FastAPI's `lifespan`
handler. Note `lifespan` -- not `@app.on_event("startup")`, which is deprecated
in current Starlette/FastAPI and will emit warnings. The models then live in
memory for the process lifetime and each request is pure forward-pass.

WHY THE SERVICE STARTS EVEN WITH NO MODELS
-------------------------------------------
If artifacts are missing we log loudly, report `status: "degraded"` on /health,
and return a clear 503 from /inspect -- rather than crashing the process.

A container that crash-loops tells an orchestrator nothing useful and hides the
real error behind a restart loop. A container that starts and honestly reports
"I am up but I have no model" is debuggable in ten seconds. That distinction is
a genuine production instinct and worth explaining if asked.

WHAT IS DELIBERATELY NOT HERE
-----------------------------
No auth, no rate limiting, no multi-tenancy -- explicitly out of scope in the
PRD. Knowing what a v1 should *not* contain is a real signal. If asked "how
would you productionise this?", that is your answer: auth, request quotas,
structured logging to a collector, model-version headers, and a canary path for
new model versions.
"""

from __future__ import annotations

import base64
import io
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import RedirectResponse
from PIL import Image, UnidentifiedImageError

from ..explain.overlay import to_png_bytes
from ..inference import InspectionEngine
from ..utils import get_device, get_logger
from .schemas import DecisionPayload, HealthResponse, InspectResponse

logger = get_logger("visionqc.api")

VERSION = "1.0.0"
RUN_DIR = os.environ.get("VISIONQC_RUN_DIR", "artifacts/synthetic")
DEVICE = os.environ.get("VISIONQC_DEVICE", "cpu")
PRIMARY = os.environ.get("VISIONQC_PRIMARY", "fusion")
# 10 MB. A cap prevents a single huge upload from exhausting memory -- the
# cheapest denial-of-service protection there is.
MAX_UPLOAD_BYTES = int(os.environ.get("VISIONQC_MAX_UPLOAD_MB", "10")) * 1024 * 1024

_engine: InspectionEngine | None = None
_load_error: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load models. Shutdown: release them."""
    global _engine, _load_error
    # One thread per worker. Without this, PyTorch spawns threads equal to core
    # count *per worker*, and on a small container they fight each other --
    # latency gets worse as you add workers, which is a genuinely confusing bug.
    torch.set_num_threads(int(os.environ.get("VISIONQC_TORCH_THREADS", "1")))
    try:
        logger.info("Loading artifacts from %s (device=%s)", RUN_DIR, DEVICE)
        _engine = InspectionEngine(Path(RUN_DIR), get_device(DEVICE), anomaly_backend="padim")
        logger.info("Ready. Docs at /docs")
    except Exception as exc:  # noqa: BLE001 - we want any failure reported, not raised
        _load_error = str(exc)
        logger.error("Startup load FAILED: %s", exc)
        logger.error("Service will run in degraded mode; /inspect returns 503.")
    yield
    _engine = None


app = FastAPI(
    title="VisionQC — Automated Visual Quality Inspection",
    description=(
        "Dual-path defect detection: a supervised classifier for known defect "
        "types, and an unsupervised anomaly detector (PaDiM) that catches "
        "deviations nobody has labelled. Returns a PASS/FAIL verdict with a "
        "heatmap showing *where* the evidence is."
    ),
    version=VERSION,
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    """Liveness plus an honest inventory of what is loaded."""
    if _engine is None:
        return HealthResponse(
            status="degraded", version=VERSION, device=DEVICE, models_loaded=[],
            classes=[], thresholds_calibrated=False, image_size=0,
        )
    loaded = []
    if _engine.has_classifier:
        loaded.append("classifier")
    if _engine.padim is not None:
        loaded.append("padim")
    if _engine.autoencoder is not None:
        loaded.append("autoencoder")
    calibrated = (
        _engine.thresholds.anomaly is not None
        or _engine.thresholds.classifier is not None
    )
    return HealthResponse(
        status="ok" if loaded else "degraded",
        version=VERSION, device=str(_engine.device), models_loaded=loaded,
        classes=_engine.classes, thresholds_calibrated=calibrated,
        image_size=_engine.image_size,
    )


@app.post(
    "/inspect",
    response_model=InspectResponse,
    tags=["inspection"],
    responses={
        400: {"description": "Unreadable or oversized image"},
        503: {"description": "Models not loaded"},
    },
)
async def inspect(
    file: UploadFile = File(..., description="Image of the part (PNG/JPEG)."),
    include_heatmap: bool = Query(True, description="Return the visual explanation."),
    primary: str | None = Query(
        None,
        description=(
            "Override the decision rule: 'fusion' (flag if either model fires), "
            "'classifier', or 'padim'. Defaults to the server setting."
        ),
    ),
) -> InspectResponse:
    if _engine is None:
        raise HTTPException(
            status_code=503,
            detail=f"Models not loaded. Startup error: {_load_error or 'unknown'}",
        )

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty upload.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({len(raw)/1e6:.1f} MB). "
                   f"Limit is {MAX_UPLOAD_BYTES/1e6:.0f} MB.",
        )

    try:
        image = Image.open(io.BytesIO(raw))
        image.load()  # force decode now, so a corrupt file fails here with a 400
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(
            status_code=400, detail=f"Could not decode image: {exc}"
        ) from exc

    # Time only the inference, not the upload -- otherwise your reported latency
    # measures the client's network, and you cannot compare runs.
    t0 = time.perf_counter()
    decision, result = _engine.inspect(image, primary=primary or PRIMARY)
    latency_ms = (time.perf_counter() - t0) * 1000

    heatmap_b64 = None
    if include_heatmap:
        try:
            panel = _engine.render_panel(result)
            heatmap_b64 = base64.b64encode(to_png_bytes(panel)).decode("ascii")
        except Exception as exc:  # noqa: BLE001
            # A rendering failure must not lose the verdict -- the decision is
            # the payload that matters; the picture is a nice-to-have.
            logger.warning("Heatmap rendering failed: %s", exc)

    return InspectResponse(
        decision=DecisionPayload(**decision.to_dict()),
        class_probabilities=result.class_probabilities,
        latency_ms=latency_ms,
        heatmap_png_base64=heatmap_b64,
    )
