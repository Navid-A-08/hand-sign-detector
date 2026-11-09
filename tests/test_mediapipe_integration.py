"""Empirical check of the flip/handedness convention with the real landmarker.

Skipped when the model or the danrasband images are not available locally.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hand_sign_detector.config import load_config

CFG = load_config()
DANRASBAND = Path.home() / ".cache/kagglehub/datasets/danrasband/asl-alphabet-test/versions/1"
SAMPLES = ["B/B0001_test.jpg", "L/L0005_test.jpg", "Y/Y0003_test.jpg", "C/C0010_test.jpg"]

pytestmark = pytest.mark.skipif(
    not CFG.paths.landmarker_model.is_file() or not DANRASBAND.is_dir(),
    reason="needs models/hand_landmarker.task and the danrasband images",
)


@pytest.fixture(scope="module")
def extractor():
    from hand_sign_detector.landmarks import HandLandmarkExtractor

    with HandLandmarkExtractor(CFG.paths.landmarker_model) as ext:
        yield ext


@pytest.mark.parametrize("name", SAMPLES)
def test_flipped_image_flips_handedness_and_keeps_features(extractor, name: str) -> None:
    import cv2

    image = cv2.imread(str(DANRASBAND / name))
    original = extractor.detect(image)
    flipped = extractor.detect(cv2.flip(image, 1))
    if original is None or flipped is None:
        pytest.skip(f"no hand detected in {name}")
    assert original.handedness != flipped.handedness
    diff = np.abs(original.features - flipped.features)
    assert float(np.median(diff)) < 0.1
