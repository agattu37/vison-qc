"""Typed configuration for VisionQC.

WHY THIS FILE EXISTS
--------------------
Beginner projects usually hard-code numbers (image size, learning rate, epochs)
inside training scripts. That is fine until you run your 12th experiment and
cannot remember which run used which settings. Then your results are not
reproducible, and "reproducible" is a word interviewers listen for.

We solve that with one small idea: every knob lives in a dataclass, every run
loads a YAML file into that dataclass, and every output directory gets a copy of
the exact config that produced it.

We use plain dataclasses instead of a heavy config framework (Hydra, etc.) on
purpose: fewer dependencies, no magic, and a reviewer can read this file in two
minutes.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    """Where the images are and how they get turned into tensors."""

    # Root of an MVTec-style folder tree. See data/README for the exact layout.
    root: str = "data/synthetic"
    # Square side length every image is resized to before it hits a model.
    # 256 is the MVTec AD convention and keeps CPU training tractable.
    image_size: int = 256
    # Fraction of the labelled `test/` folder reserved for final evaluation.
    # Both the supervised and unsupervised paths are scored on this same set.
    holdout_frac: float = 0.5
    # Fraction of the *remaining* labelled data used for validation.
    val_frac: float = 0.2
    # DataLoader workers. 0 is safest on Windows and inside slim containers.
    num_workers: int = 2
    # ImageNet statistics. The pretrained backbone was trained with these, so
    # deviating from them throws away part of the transfer-learning benefit.
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass
class ClassifierConfig:
    """Supervised defect classifier (transfer learning)."""

    backbone: str = "resnet18"
    pretrained: bool = True
    # Freeze everything except the final block + head for the first N epochs.
    # With a few hundred labelled images, full fine-tuning from step 1 overfits.
    freeze_epochs: int = 2
    epochs: int = 12
    batch_size: int = 16
    lr_head: float = 1e-3
    lr_backbone: float = 1e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    # Inverse-frequency class weights in the loss. Defect classes are rare, and
    # without this the model can score well by predicting "good" every time.
    use_class_weights: bool = True
    dropout: float = 0.2
    early_stop_patience: int = 5


@dataclass
class AutoencoderConfig:
    """Unsupervised baseline: convolutional autoencoder."""

    latent_channels: int = 64
    base_channels: int = 32
    epochs: int = 30
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-5
    # Gaussian blur applied to the per-pixel error map. Single-pixel noise is
    # not a defect; blurring makes the score respond to *regions*.
    smooth_sigma: float = 4.0


@dataclass
class PadimConfig:
    """Unsupervised upgrade: PaDiM (patch distribution modelling)."""

    backbone: str = "resnet18"
    pretrained: bool = True
    # Which residual stages to concatenate. Early layers carry texture, later
    # layers carry shape/semantics; PaDiM's whole point is combining both.
    layers: tuple[str, ...] = ("layer1", "layer2", "layer3")
    # Random projection down to this many channels. The paper shows 100 random
    # dimensions match the full 448 while cutting covariance cost ~20x.
    n_components: int = 100
    # Patch grid side. 32 -> 1024 patches at 256px input. Bigger = sharper maps
    # but memory grows as grid^2 * d^2.
    embed_grid: int = 32
    # Ridge term added to each covariance diagonal so the inverse is stable.
    reg: float = 0.01
    smooth_sigma: float = 4.0
    seed: int = 1337


@dataclass
class DecisionConfig:
    """Turning model scores into a PASS/FAIL call."""

    # Relative business cost of letting a defect ship vs. scrapping a good part.
    # 10.0 means "one escaped defect hurts as much as ten false alarms".
    cost_false_negative: float = 10.0
    cost_false_positive: float = 1.0
    # Which signal drives the final call: "padim", "autoencoder",
    # "classifier", or "fusion" (flag if *either* path fires).
    primary: str = "padim"
    # Filled in by the threshold-selection step; None means "not calibrated yet".
    classifier_threshold: float | None = None
    anomaly_threshold: float | None = None


@dataclass
class Config:
    seed: int = 42
    device: str = "auto"  # "auto" | "cpu" | "cuda" | "mps"
    output_dir: str = "artifacts"
    run_name: str = "default"
    data: DataConfig = field(default_factory=DataConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    autoencoder: AutoencoderConfig = field(default_factory=AutoencoderConfig)
    padim: PadimConfig = field(default_factory=PadimConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)

    # ---- serialisation helpers -------------------------------------------
    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Config":
        """Build a Config from a nested dict, ignoring unknown keys loudly."""
        sections = {
            "data": DataConfig,
            "classifier": ClassifierConfig,
            "autoencoder": AutoencoderConfig,
            "padim": PadimConfig,
            "decision": DecisionConfig,
        }
        kwargs: dict[str, Any] = {}
        for key, value in raw.items():
            if key in sections:
                kwargs[key] = _build(sections[key], value or {})
            elif key in {f.name for f in dataclasses.fields(cls)}:
                kwargs[key] = value
            else:
                raise ValueError(
                    f"Unknown config key '{key}'. Typo? Valid top-level keys: "
                    f"{sorted({f.name for f in dataclasses.fields(cls)})}"
                )
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)

    @property
    def run_dir(self) -> Path:
        """All artifacts for one run live under artifacts/<run_name>/."""
        return Path(self.output_dir) / self.run_name


def _build(cls: type, value: dict[str, Any]):
    valid = {f.name for f in dataclasses.fields(cls)}
    unknown = set(value) - valid
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    # YAML gives lists; dataclass fields typed as tuple should stay tuples.
    coerced = {
        k: tuple(v) if isinstance(v, list) else v for k, v in value.items()
    }
    return cls(**coerced)
