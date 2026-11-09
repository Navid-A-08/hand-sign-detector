"""Camera-free inference logic: thresholding, top-k and temporal smoothing."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass

import numpy as np

UNCERTAIN = "Uncertain"
NO_HAND = "nothing"


@dataclass(frozen=True)
class Prediction:
    """One frame's prediction."""

    label: str
    confidence: float
    top_k: list[tuple[str, float]]

    @property
    def is_confident(self) -> bool:
        """True when the label is a real class rather than 'Uncertain'."""
        return self.label != UNCERTAIN


def top_k(probs: np.ndarray, classes: list[str], k: int = 2) -> list[tuple[str, float]]:
    """Return the ``k`` most likely ``(label, probability)`` pairs, best first."""
    probs = np.asarray(probs, dtype=np.float64).ravel()
    if probs.size != len(classes):
        raise ValueError("probs and classes must have the same length")
    order = np.argsort(-probs, kind="stable")[: max(1, k)]
    return [(classes[i], float(probs[i])) for i in order]


def apply_threshold(probs: np.ndarray, classes: list[str], threshold: float = 0.60, k: int = 2) -> Prediction:
    """Pick the top class, or 'Uncertain' if its probability is below ``threshold``."""
    ranked = top_k(probs, classes, k)
    best_label, best_prob = ranked[0]
    label = best_label if best_prob >= threshold else UNCERTAIN
    return Prediction(label=label, confidence=best_prob, top_k=ranked)


def no_hand_prediction() -> Prediction:
    """Prediction used when MediaPipe finds no hand."""
    return Prediction(label=NO_HAND, confidence=1.0, top_k=[(NO_HAND, 1.0)])


class MajorityVoteSmoother:
    """Majority vote over the last ``window`` frame labels.

    Ties are broken in favour of the most recent label among the tied ones,
    so the display reacts to a change as soon as it leads.
    """

    def __init__(self, window: int = 10, min_votes: int = 1) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        if not 1 <= min_votes <= window:
            raise ValueError("min_votes must be in [1, window]")
        self.window = window
        self.min_votes = min_votes
        self._labels: deque[str] = deque(maxlen=window)

    def update(self, label: str) -> str:
        """Add one frame's label and return the smoothed label."""
        self._labels.append(label)
        return self.current()

    def current(self) -> str:
        """Return the current smoothed label ('Uncertain' if no label has ``min_votes``)."""
        if not self._labels:
            return UNCERTAIN
        counts = Counter(self._labels)
        best = max(counts.values())
        if best < self.min_votes:
            return UNCERTAIN
        tied = {label for label, count in counts.items() if count == best}
        for label in reversed(self._labels):
            if label in tied:
                return label
        return UNCERTAIN

    def reset(self) -> None:
        """Forget all history."""
        self._labels.clear()

    def __len__(self) -> int:
        return len(self._labels)
