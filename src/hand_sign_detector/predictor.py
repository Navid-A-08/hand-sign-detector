"""End-to-end predictor: image -> landmarks -> classifier -> thresholded prediction."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .inference import Prediction, apply_threshold, no_hand_prediction
from .landmarks import HandDetection, HandLandmarkExtractor
from .models import load_model, predict_proba

logger = logging.getLogger(__name__)

# MediaPipe hand skeleton, used for drawing.
HAND_CONNECTIONS: list[tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]  # fmt: skip


class SignPredictor:
    """Classify a hand sign from a BGR image or a pre-computed feature vector."""

    def __init__(
        self,
        model_path: Path | str,
        extractor: HandLandmarkExtractor | None = None,
        threshold: float = 0.60,
        k: int = 2,
    ) -> None:
        self.model, self.metadata = load_model(Path(model_path))
        self.extractor = extractor
        self.threshold = threshold
        self.k = k

    def predict_features(self, features: np.ndarray) -> Prediction:
        """Predict from a normalized 63-D feature vector."""
        probs, classes = predict_proba(self.model, features)
        return apply_threshold(probs[0], classes, self.threshold, self.k)

    def predict_image(self, image_bgr: np.ndarray, timestamp_ms: int | None = None) -> tuple[Prediction, HandDetection | None]:
        """Detect a hand and classify it. Returns 'nothing' when no hand is found."""
        if self.extractor is None:
            raise RuntimeError("SignPredictor needs a HandLandmarkExtractor to predict from images")
        detection = self.extractor.detect(image_bgr, timestamp_ms)
        if detection is None:
            return no_hand_prediction(), None
        try:
            features = detection.features
        except ValueError:
            logger.debug("Degenerate landmarks; treating as no hand")
            return no_hand_prediction(), None
        return self.predict_features(features), detection


def draw_hand(image_bgr: np.ndarray, detection: HandDetection) -> None:
    """Draw the landmark skeleton in place on an un-mirrored frame."""
    import cv2

    pts = detection.points_px[:, :2].astype(int)
    for a, b in HAND_CONNECTIONS:
        cv2.line(image_bgr, tuple(pts[a]), tuple(pts[b]), (0, 200, 0), 2)
    for x, y in pts:
        cv2.circle(image_bgr, (int(x), int(y)), 3, (0, 0, 255), -1)
