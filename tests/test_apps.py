"""Import checks and camera-free logic for webcam.py and collect.py."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from hand_sign_detector.collection import CollectionSession, validate_signer_id
from hand_sign_detector.inference import Prediction
from hand_sign_detector.landmarks import HandDetection

LABELS = ["A", "B", "nothing"]


@pytest.mark.parametrize("module", ["webcam", "collect", "predict_image", "train", "evaluate", "prepare_data"])
def test_scripts_import(module: str) -> None:
    assert hasattr(importlib.import_module(module), "main")


def test_key_to_label() -> None:
    collect = importlib.import_module("collect")
    letters = ["A", "Q", "Y"]
    assert collect.key_to_label(ord("q"), letters, "nothing") == "Q"
    assert collect.key_to_label(ord("0"), letters, "nothing") == "nothing"
    assert collect.key_to_label(ord("j"), letters, "nothing") is None
    assert collect.key_to_label(255, letters, "nothing") is None


def test_format_overlay_shows_top2() -> None:
    webcam = importlib.import_module("webcam")
    pred = Prediction(label="Uncertain", confidence=0.45, top_k=[("M", 0.45), ("N", 0.40)])
    lines = webcam.format_overlay("M", pred, 29.6)
    assert lines[0] == "Sign: M"
    assert lines[1] == "1. M 45%"
    assert lines[2] == "2. N 40%"


@pytest.mark.parametrize("bad", ["", "has space", "x" * 40, "../etc"])
def test_signer_id_validation(bad: str) -> None:
    with pytest.raises(ValueError):
        validate_signer_id(bad)


def test_collection_session_roundtrip(tmp_path, hand_points: np.ndarray) -> None:
    session = CollectionSession("navid", LABELS, session_id="s1")
    detection = HandDetection(points_px=hand_points, handedness="Left", handedness_score=0.9)
    assert session.add("A", detection, 1) is True
    assert session.add("A", None, 2) is False
    assert session.add("B", detection, 3) is True
    with pytest.raises(ValueError):
        session.add("J", detection, 4)
    assert session.counts() == {"A": 1, "B": 1}
    path = session.save(tmp_path)
    import pandas as pd

    frame = pd.read_parquet(path)
    assert len(frame) == 3
    assert set(frame["signer_id"]) == {"navid"}
    assert frame["group"].iloc[0] == "webcam:navid:s1"
    assert frame["detected"].tolist() == [True, False, True]
    np.testing.assert_allclose(frame.iloc[0, 8:].to_numpy(dtype=np.float32), detection.features, atol=1e-6)
