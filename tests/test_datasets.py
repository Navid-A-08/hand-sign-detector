"""File-name attribution, label mapping and metric bookkeeping."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from hand_sign_detector.config import load_config
from hand_sign_detector.datasets import FEATURE_COLUMNS, classify_sau_file, map_label
from hand_sign_detector.evaluation import evaluate_frame, top_confused_pairs

LETTERS = load_config().classes.letters


@pytest.mark.parametrize(
    "name,source",
    [
        ("A123.jpg", "grassknoted"),
        ("nothing2999.jpg", "grassknoted"),
        ("A0001_test.jpg", "danrasband"),
        ("A (12).jpg", "sau_paren"),
        ("h_62_rotate_4.jpeg", "sau_rotate"),
        ("19.jpg", "sau_numeric"),
    ],
)
def test_classify_sau_file(name: str, source: str) -> None:
    assert classify_sau_file(name)[0] == source


def test_rotated_copies_share_a_group() -> None:
    assert classify_sau_file("h_62_rotate_1.jpeg")[1] == classify_sau_file("H_62_rotate_4.jpeg")[1]


def test_unknown_pattern() -> None:
    assert classify_sau_file("weird-name.png") is None


@pytest.mark.parametrize("folder,label", [("a", "A"), ("Y", "Y"), ("nothing", "nothing"), ("J", None), ("Z", None), ("del", None), ("space", None)])
def test_map_label(folder: str, label: str | None) -> None:
    assert map_label(folder, LETTERS, "nothing") == label


def test_config_has_24_letters_plus_nothing() -> None:
    cfg = load_config()
    assert len(cfg.classes.letters) == 24
    assert cfg.classes.labels[-1] == "nothing"
    assert 0.55 <= cfg.inference.confidence_threshold <= 0.65


def _frame(labels: list[str], detected: list[bool]) -> pd.DataFrame:
    rows = []
    for label, det in zip(labels, detected):
        row = {"label": label, "detected": det}
        row.update(zip(FEATURE_COLUMNS, (np.zeros(63) if det else np.full(63, np.nan)).tolist()))
        rows.append(row)
    return pd.DataFrame(rows)


def test_detection_rate_is_separate_from_accuracy() -> None:
    model = DummyClassifier(strategy="constant", constant="A").fit(np.zeros((2, 63)), ["A", "B"])
    frame = _frame(["A", "A", "B", "A", "nothing"], [True, True, True, False, False])
    metrics = evaluate_frame(model, frame, ["A", "B", "nothing"], "nothing")
    assert metrics["detection_rate_letters"] == pytest.approx(3 / 4)
    assert metrics["classification_accuracy"] == pytest.approx(2 / 3)
    # Correct: two detected A's and the undetected 'nothing' image.
    assert metrics["end_to_end_accuracy"] == pytest.approx(3 / 5)


def test_top_confused_pairs() -> None:
    cm = np.array([[5, 3], [1, 9]])
    pairs = top_confused_pairs(cm, ["M", "N"])
    assert pairs[0] == {"true": "M", "pred": "N", "count": 3, "share_of_true": 0.375}


def test_near_duplicate_mask_catches_resized_copy(tmp_path) -> None:
    """A resized, re-encoded copy is flagged; an unrelated image is not (unit-test fixtures only)."""
    import cv2

    from hand_sign_detector.datasets import near_duplicate_mask

    rng = np.random.default_rng(1)
    base = cv2.GaussianBlur(rng.integers(0, 255, (400, 400, 3), dtype=np.uint8), (31, 31), 0)
    other = cv2.GaussianBlur(rng.integers(0, 255, (400, 400, 3), dtype=np.uint8), (31, 31), 0)
    cv2.imwrite(str(tmp_path / "ref.png"), cv2.resize(base, (200, 200)))
    cv2.imwrite(str(tmp_path / "copy.jpg"), base, [cv2.IMWRITE_JPEG_QUALITY, 70])
    cv2.imwrite(str(tmp_path / "other.png"), other)
    mask = near_duplicate_mask([str(tmp_path / "copy.jpg"), str(tmp_path / "other.png")], [str(tmp_path / "ref.png")], workers=1)
    assert mask.tolist() == [True, False]
