"""Metrics, confusion matrices and split helpers.

Detection rate (did MediaPipe find a hand?) is always reported separately from
classification accuracy (was the letter right, given a detected hand?).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from .datasets import FEATURE_COLUMNS

logger = logging.getLogger(__name__)


def filter_labels(frame: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    """Keep only rows whose label is in ``labels``."""
    return frame[frame["label"].isin(labels)].reset_index(drop=True)


def detected_xy(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return features and labels for rows where a hand was detected."""
    part = frame[frame["detected"]]
    return part[FEATURE_COLUMNS].to_numpy(dtype=np.float32), part["label"].to_numpy()


def top_confused_pairs(cm: np.ndarray, labels: list[str], n: int = 10) -> list[dict]:
    """Return the ``n`` largest off-diagonal cells as true -> predicted pairs."""
    pairs: list[dict] = []
    totals = cm.sum(axis=1)
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            if i != j and cm[i, j] > 0:
                share = float(cm[i, j] / totals[i]) if totals[i] else 0.0
                pairs.append({"true": true_label, "pred": pred_label, "count": int(cm[i, j]), "share_of_true": round(share, 4)})
    pairs.sort(key=lambda p: (-p["count"], -p["share_of_true"]))
    return pairs[:n]


def evaluate_frame(model: ClassifierMixin, frame: pd.DataFrame, labels: list[str], nothing_label: str) -> dict:
    """Evaluate a fitted model on a landmark table.

    Metrics:
        * ``detection_rate_letters``: share of letter images where a hand was found.
        * ``classification_accuracy``: accuracy on letter images with a detected hand.
        * ``end_to_end_accuracy``: accuracy over all images, where 'no hand' counts as
          predicting 'nothing' (right for 'nothing' images, wrong for letters).

    Returns:
        A JSON-serializable dict including the confusion matrix on detected samples.
    """
    letters = [label for label in labels if label != nothing_label]
    letter_rows = frame[frame["label"].isin(letters)]
    nothing_rows = frame[frame["label"] == nothing_label]
    metrics: dict = {
        "n_images": int(len(frame)),
        "n_letter_images": int(len(letter_rows)),
        "n_nothing_images": int(len(nothing_rows)),
        "detection_rate_letters": float(letter_rows["detected"].mean()) if len(letter_rows) else None,
        "detection_rate_nothing_images": float(nothing_rows["detected"].mean()) if len(nothing_rows) else None,
    }

    predictions = np.full(len(frame), nothing_label, dtype=object)
    mask = frame["detected"].to_numpy(dtype=bool)
    if mask.any():
        x = frame.loc[mask, FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        predictions[mask] = model.predict(x)
    truth = frame["label"].to_numpy()
    letter_mask = np.isin(truth, letters)
    detected_letters = mask & letter_mask

    metrics["n_classified"] = int(detected_letters.sum())
    metrics["classification_accuracy"] = (
        float(accuracy_score(truth[detected_letters], predictions[detected_letters])) if detected_letters.any() else None
    )
    metrics["classification_macro_f1"] = (
        float(f1_score(truth[detected_letters], predictions[detected_letters], labels=letters, average="macro", zero_division=0))
        if detected_letters.any()
        else None
    )
    metrics["end_to_end_accuracy"] = float(accuracy_score(truth, predictions)) if len(frame) else None
    present = [label for label in labels if label in set(truth) | set(predictions[mask])]
    cm = confusion_matrix(truth[mask], predictions[mask], labels=present) if mask.any() else np.zeros((0, 0), dtype=int)
    metrics["confusion_labels"] = present
    metrics["confusion_matrix"] = cm.tolist()
    metrics["top_confused_pairs"] = top_confused_pairs(cm, present)
    per_class = {}
    for label in letters:
        sel = detected_letters & (truth == label)
        if sel.any():
            per_class[label] = round(float((predictions[sel] == label).mean()), 4)
    metrics["per_class_accuracy"] = per_class
    return metrics


def save_confusion(metrics: dict, stem: Path, title: str) -> None:
    """Save the confusion matrix from ``metrics`` as ``<stem>.csv`` and ``<stem>.png``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = metrics["confusion_labels"]
    cm = np.asarray(metrics["confusion_matrix"])
    if cm.size == 0:
        logger.warning("Empty confusion matrix for %s", stem)
        return
    stem.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cm, index=labels, columns=labels).rename_axis("true\\pred").to_csv(stem.with_suffix(".csv"))
    row_sums = cm.sum(axis=1, keepdims=True)
    normalized = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums > 0)
    fig, ax = plt.subplots(figsize=(11, 9))
    image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=90)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    acc = metrics.get("classification_accuracy")
    det = metrics.get("detection_rate_letters")
    subtitle = f"cls acc {acc:.3f} | detection {det:.3f}" if acc is not None and det is not None else ""
    ax.set_title(f"{title}\n{subtitle} (row-normalized, detected hands only)")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"), dpi=110)
    plt.close(fig)


def write_json(data: dict, path: Path) -> None:
    """Write ``data`` as pretty JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def summarize(metrics: dict) -> dict:
    """The headline numbers of a metrics dict (no matrices)."""
    keys = [
        "n_images",
        "n_classified",
        "detection_rate_letters",
        "detection_rate_nothing_images",
        "classification_accuracy",
        "classification_macro_f1",
        "end_to_end_accuracy",
    ]
    out = {key: metrics.get(key) for key in keys}
    out["top_confused_pairs"] = metrics.get("top_confused_pairs", [])[:5]
    return out
