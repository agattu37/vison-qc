# VisionQC serving image.
#
# WHY A MULTI-STAGE BUILD
# -----------------------
# Stage 1 installs dependencies (slow, cached). Stage 2 copies only the
# installed packages and our source into a clean image. Build tools, pip caches
# and compilers never reach the final layer.
#
# WHY THE CPU-ONLY TORCH INDEX
# ----------------------------
# The default PyPI `torch` wheel bundles CUDA libraries: roughly 2.5 GB of
# NVIDIA runtime you cannot use on a CPU host. Pulling from PyTorch's CPU index
# gives a ~200 MB wheel instead. On a free hosting tier with an image-size cap,
# this single line is the difference between deploying and not deploying.

# ---------- stage 1: builder ----------
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build
COPY requirements-serve.txt .

# --extra-index-url, not --index-url: torch and torchvision come from the CPU
# index, everything else still resolves from PyPI.
RUN pip install --prefix=/install \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements-serve.txt

# ---------- stage 2: runtime ----------
FROM python:3.12-slim

# Run as a non-root user. If the container is ever compromised, the attacker
# lands as an unprivileged account rather than root. Cheap, and the first thing
# a security review looks for.
RUN useradd --create-home --shell /bin/bash visionqc

COPY --from=builder /install /usr/local

WORKDIR /app
COPY --chown=visionqc:visionqc src/ ./src/
COPY --chown=visionqc:visionqc configs/ ./configs/

# Artifacts are baked in when present. Mount them at runtime instead if you
# prefer to keep 50 MB checkpoints out of your image:
#   docker run -v $(pwd)/artifacts:/app/artifacts ...
COPY --chown=visionqc:visionqc artifacts/ ./artifacts/

ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    VISIONQC_RUN_DIR=/app/artifacts/synthetic \
    VISIONQC_DEVICE=cpu \
    VISIONQC_PRIMARY=fusion \
    VISIONQC_TORCH_THREADS=1

USER visionqc
EXPOSE 8000

# HEALTHCHECK lets Docker/orchestrators know the difference between "process is
# running" and "app is actually serving". Without it a hung worker looks healthy.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

# One worker. Each worker loads its own full copy of the models into RAM, so on
# a 512 MB free tier a second worker means an out-of-memory kill, not throughput.
CMD ["uvicorn", "visionqc.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
