"""Typed configuration loaded from ``config.yaml`` and validated at startup."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"


@dataclass(frozen=True)
class PathsConfig:
    """Filesystem locations, resolved against the repo root."""

    data_dir: Path
    landmarks_dir: Path
    models_dir: Path
    reports_dir: Path
    landmarker_model: Path


@dataclass(frozen=True)
class LandmarkerConfig:
    """MediaPipe Hand Landmarker settings."""

    model_url: str
    num_hands: int = 1
    min_detection_confidence: float = 0.5
    min_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


@dataclass(frozen=True)
class ClassesConfig:
    """Label set: 24 static letters plus a 'nothing' class."""

    letters: list[str]
    nothing_label: str = "nothing"

    @property
    def labels(self) -> list[str]:
        """All class labels in a fixed order."""
        return [*self.letters, self.nothing_label]


@dataclass(frozen=True)
class TrainingConfig:
    """Training hyperparameters and data filters."""

    seed: int = 42
    exclude_sources: list[str] = field(default_factory=list)
    max_per_group_label: int = 0
    mlp: dict = field(default_factory=dict)
    baseline: dict = field(default_factory=dict)


@dataclass(frozen=True)
class InferenceConfig:
    """Live and single-image inference settings."""

    confidence_threshold: float = 0.60
    smoothing_window: int = 10
    top_k: int = 2


@dataclass(frozen=True)
class AppConfig:
    """Top-level configuration."""

    paths: PathsConfig
    landmarker: LandmarkerConfig
    classes: ClassesConfig
    training: TrainingConfig
    inference: InferenceConfig


def _validate(cfg: AppConfig) -> None:
    """Fail fast on obviously bad configuration."""
    letters = cfg.classes.letters
    if len(set(letters)) != len(letters):
        raise ValueError("Duplicate letters in classes.letters")
    if {"J", "Z"} & set(letters):
        raise ValueError("J and Z are dynamic signs and are not supported in v1")
    if not 0.0 < cfg.inference.confidence_threshold < 1.0:
        raise ValueError("inference.confidence_threshold must be in (0, 1)")
    if cfg.inference.smoothing_window < 1:
        raise ValueError("inference.smoothing_window must be >= 1")
    if cfg.inference.top_k < 1:
        raise ValueError("inference.top_k must be >= 1")


def load_config(path: Path | str | None = None) -> AppConfig:
    """Load and validate the YAML config.

    Args:
        path: Optional path to a YAML file. Defaults to ``config.yaml`` at the repo root.

    Returns:
        The validated configuration.
    """
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    paths = {key: (REPO_ROOT / value).resolve() for key, value in raw["paths"].items()}
    cfg = AppConfig(
        paths=PathsConfig(**paths),
        landmarker=LandmarkerConfig(**raw["landmarker"]),
        classes=ClassesConfig(**raw["classes"]),
        training=TrainingConfig(**raw.get("training", {})),
        inference=InferenceConfig(**raw.get("inference", {})),
    )
    _validate(cfg)
    return cfg
