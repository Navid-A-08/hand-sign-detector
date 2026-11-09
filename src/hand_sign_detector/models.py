"""Classifier registry: a small MLP and a random-forest / SVM baseline."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

logger = logging.getLogger(__name__)

ModelFactory = Callable[[dict, int], Pipeline]
MODEL_REGISTRY: dict[str, ModelFactory] = {}


def register(name: str) -> Callable[[ModelFactory], ModelFactory]:
    """Register a model factory under ``name``."""

    def decorator(factory: ModelFactory) -> ModelFactory:
        MODEL_REGISTRY[name] = factory
        return factory

    return decorator


@register("mlp")
def build_mlp(params: dict, seed: int) -> Pipeline:
    """Standardize features, then a two-layer MLP."""
    clf = MLPClassifier(
        hidden_layer_sizes=tuple(params.get("hidden_layer_sizes", (128, 64))),
        alpha=float(params.get("alpha", 1e-4)),
        max_iter=int(params.get("max_iter", 200)),
        early_stopping=bool(params.get("early_stopping", True)),
        random_state=seed,
    )
    return make_pipeline(StandardScaler(), clf)


@register("random_forest")
def build_random_forest(params: dict, seed: int) -> Pipeline:
    """Random-forest baseline (no scaling needed, kept in a pipeline for a uniform API)."""
    clf = RandomForestClassifier(
        n_estimators=int(params.get("n_estimators", 300)),
        min_samples_leaf=int(params.get("min_samples_leaf", 1)),
        n_jobs=-1,
        random_state=seed,
    )
    return make_pipeline(clf)


@register("svm")
def build_svm(params: dict, seed: int) -> Pipeline:
    """RBF SVM baseline with probability outputs (slow on >50k samples)."""
    clf = SVC(C=float(params.get("C", 10.0)), gamma=params.get("gamma", "scale"), probability=True, random_state=seed)
    return make_pipeline(StandardScaler(), clf)


def build_model(name: str, params: dict, seed: int) -> Pipeline:
    """Build a registered model by name."""
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Known: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](params, seed)


def save_model(model: ClassifierMixin, path: Path, metadata: dict) -> None:
    """Persist a fitted model together with its metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "metadata": metadata}, path, compress=3)
    logger.info("Saved model to %s (%.1f MB)", path, path.stat().st_size / 1e6)


def load_model(path: Path) -> tuple[ClassifierMixin, dict]:
    """Load a model saved with :func:`save_model`."""
    bundle = joblib.load(path)
    return bundle["model"], bundle["metadata"]


def predict_proba(model: ClassifierMixin, features: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Return class probabilities and the class order."""
    probs = model.predict_proba(np.atleast_2d(features))
    return probs, [str(c) for c in model.classes_]
