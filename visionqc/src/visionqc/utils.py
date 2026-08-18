"""Small shared helpers: seeding, device choice, logging, timing, JSON I/O.

WHY THIS FILE EXISTS
--------------------
Three things quietly ruin ML projects, and all three are fixed here in ~100
lines:

1. **Irreproducibility.** You get 0.94 AUROC, rerun, get 0.89, and cannot tell
   whether your change helped. `set_seed` pins every random source.
2. **Device drift.** Code that only runs on your GPU laptop fails in the Docker
   container. `get_device` resolves this once, centrally.
3. **Silent runs.** A training script that prints nothing for 20 minutes is
   impossible to debug. `get_logger` gives timestamped, level-tagged output.
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Pin every RNG we touch.

    Note the honest caveat: `deterministic=True` makes cuDNN pick reproducible
    algorithms, which can be slower. On CPU it costs nothing. We accept the
    tradeoff because being able to trust a comparison matters more here than a
    few percent of throughput.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(preference: str = "auto") -> torch.device:
    """Resolve 'auto' to the best available accelerator.

    Order: CUDA (NVIDIA) -> MPS (Apple Silicon) -> CPU. Everything in this repo
    is written to run correctly on CPU, just slower; the API deliberately
    targets CPU because that is what free hosting tiers give you.
    """
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_logger(name: str = "visionqc", level: int = logging.INFO) -> logging.Logger:
    """A logger that prints once, to stdout, with timestamps."""
    logger = logging.getLogger(name)
    if not logger.handlers:  # guard against duplicate handlers on re-import
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(level)
    return logger


@contextmanager
def timer(label: str, logger: logging.Logger | None = None) -> Iterator[dict[str, float]]:
    """Measure wall-clock time for a block.

    Used for the latency requirement in the PRD (<2s per image on CPU). Yields a
    dict so the caller can read the elapsed time after the block exits.
    """
    result: dict[str, float] = {}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["seconds"] = time.perf_counter() - start
        msg = f"{label}: {result['seconds']:.3f}s"
        (logger.info(msg) if logger else print(msg))


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_json_default)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _json_default(o: Any) -> Any:
    """Let json.dump handle numpy scalars/arrays and Paths without complaining."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Not JSON serialisable: {type(o)}")


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """Return (total, trainable) parameter counts.

    Worth printing in every training log: it is the fastest way to catch a
    freezing bug, where you *think* the backbone is frozen but it is not.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
