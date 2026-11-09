"""Thresholding, top-k and majority-vote smoothing used by webcam.py."""

from __future__ import annotations

import numpy as np
import pytest

from hand_sign_detector.inference import (
    UNCERTAIN,
    MajorityVoteSmoother,
    apply_threshold,
    no_hand_prediction,
    top_k,
)

CLASSES = ["A", "B", "C"]


def test_top_k_order() -> None:
    assert top_k(np.array([0.2, 0.5, 0.3]), CLASSES, 2) == [("B", 0.5), ("C", 0.3)]


def test_top_k_length_mismatch() -> None:
    with pytest.raises(ValueError):
        top_k(np.array([0.5, 0.5]), CLASSES)


def test_threshold_accepts_confident() -> None:
    pred = apply_threshold(np.array([0.1, 0.75, 0.15]), CLASSES, threshold=0.6)
    assert pred.label == "B"
    assert pred.is_confident
    assert [label for label, _ in pred.top_k] == ["B", "C"]


def test_threshold_rejects_below() -> None:
    pred = apply_threshold(np.array([0.3, 0.45, 0.25]), CLASSES, threshold=0.6)
    assert pred.label == UNCERTAIN
    assert not pred.is_confident
    assert pred.top_k[0] == ("B", 0.45)


def test_threshold_boundary_is_inclusive() -> None:
    assert apply_threshold(np.array([0.6, 0.4, 0.0]), CLASSES, threshold=0.6).label == "A"


def test_no_hand_prediction() -> None:
    assert no_hand_prediction().label == "nothing"


def test_smoother_majority() -> None:
    smoother = MajorityVoteSmoother(window=5)
    for label in ["A", "A", "B", "A", "C"]:
        out = smoother.update(label)
    assert out == "A"


def test_smoother_window_forgets_old_frames() -> None:
    smoother = MajorityVoteSmoother(window=3)
    for label in ["A", "A", "A", "B", "B"]:
        out = smoother.update(label)
    assert out == "B"
    assert len(smoother) == 3


def test_smoother_tie_prefers_most_recent() -> None:
    smoother = MajorityVoteSmoother(window=4)
    for label in ["A", "A", "B", "B"]:
        out = smoother.update(label)
    assert out == "B"


def test_smoother_suppresses_single_frame_flicker() -> None:
    smoother = MajorityVoteSmoother(window=5)
    outputs = [smoother.update(label) for label in ["A", "A", "A", "S", "A", "A"]]
    assert "S" not in outputs


def test_smoother_min_votes_and_reset() -> None:
    smoother = MajorityVoteSmoother(window=4, min_votes=3)
    smoother.update("A")
    smoother.update("B")
    assert smoother.current() == UNCERTAIN
    smoother.reset()
    assert smoother.current() == UNCERTAIN


@pytest.mark.parametrize("window,min_votes", [(0, 1), (3, 0), (3, 4)])
def test_smoother_invalid_args(window: int, min_votes: int) -> None:
    with pytest.raises(ValueError):
        MajorityVoteSmoother(window=window, min_votes=min_votes)
