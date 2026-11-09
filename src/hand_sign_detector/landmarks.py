"""Hand landmark extraction and normalization.

Conventions (fixed for training and inference):

* Images are passed to MediaPipe **as captured**, never mirrored. ``webcam.py``
  flips frames only for display, after detection.
* Landmarks are converted to pixel-aspect units (x * width, y * height,
  z * width) so non-square frames are not distorted.
* The wrist (landmark 0) is moved to the origin and coordinates are divided by
  the palm size, the 3-D distance from the wrist to the middle-finger MCP
  (landmark 9).
* Hands that MediaPipe labels ``"Left"`` are mirrored (x -> -x) into a
  canonical right hand. Hands with unknown handedness are left as they are.
* Features are **not** rotation-normalized, because some letters (H/U, G/Q,
  K/P) differ mainly in orientation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

import numpy as np

logger = logging.getLogger(__name__)

NUM_LANDMARKS = 21
FEATURE_DIM = NUM_LANDMARKS * 3
WRIST = 0
MIDDLE_MCP = 9
CANONICAL_HANDEDNESS = "Right"
_MIN_PALM_SIZE = 1e-6


def to_pixel_space(points: np.ndarray, width: float, height: float) -> np.ndarray:
    """Convert MediaPipe normalized image landmarks to pixel-aspect units.

    Args:
        points: Array of shape (21, 3) with x, y in [0, 1] and z on the x scale.
        width: Image width in pixels (or 1.0 when unknown).
        height: Image height in pixels (or 1.0 when unknown).

    Returns:
        Array of shape (21, 3) in pixel-aspect units.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(NUM_LANDMARKS, 3)
    return pts * np.array([width, height, width], dtype=np.float64)


def mirror_x(points: np.ndarray) -> np.ndarray:
    """Mirror landmarks horizontally about the origin (x -> -x)."""
    out = np.array(points, dtype=np.float64, copy=True).reshape(NUM_LANDMARKS, 3)
    out[:, 0] *= -1.0
    return out


def palm_size(points: np.ndarray) -> float:
    """Return the 3-D wrist-to-middle-MCP distance."""
    pts = np.asarray(points, dtype=np.float64).reshape(NUM_LANDMARKS, 3)
    return float(np.linalg.norm(pts[MIDDLE_MCP] - pts[WRIST]))


def normalize_landmarks(points: np.ndarray, handedness: str | None = None) -> np.ndarray:
    """Turn 21 raw landmarks into the canonical 63-D feature vector.

    Args:
        points: Array of shape (21, 3) or (63,), in pixel-aspect units.
        handedness: MediaPipe handedness label ("Left"/"Right") or None if unknown.

    Returns:
        A float32 vector of length 63 (x0, y0, z0, x1, ...).

    Raises:
        ValueError: If the input has the wrong shape, is not finite, or the palm size is ~0.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.size != FEATURE_DIM:
        raise ValueError(f"Expected {FEATURE_DIM} values, got {pts.size}")
    pts = pts.reshape(NUM_LANDMARKS, 3)
    if not np.all(np.isfinite(pts)):
        raise ValueError("Landmarks contain NaN or inf")
    centered = pts - pts[WRIST]
    scale = palm_size(centered)
    if scale < _MIN_PALM_SIZE:
        raise ValueError("Palm size is ~0; landmarks are degenerate")
    normalized = centered / scale
    if handedness is not None and handedness != CANONICAL_HANDEDNESS:
        normalized = mirror_x(normalized)
    return normalized.reshape(-1).astype(np.float32)


@dataclass(frozen=True)
class HandDetection:
    """One detected hand."""

    points_px: np.ndarray
    handedness: str
    handedness_score: float

    @property
    def features(self) -> np.ndarray:
        """Canonical 63-D feature vector for this hand."""
        return normalize_landmarks(self.points_px, self.handedness)


class HandLandmarkExtractor:
    """Thin wrapper around the MediaPipe Tasks Hand Landmarker.

    Use as a context manager so the native graph is closed cleanly.
    """

    def __init__(
        self,
        model_path: Path | str,
        video_mode: bool = False,
        num_hands: int = 1,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        from mediapipe.tasks.python import BaseOptions, vision

        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Hand landmarker model not found at {model_path}. Run: python prepare_data.py --model-only"
            )
        running_mode = vision.RunningMode.VIDEO if video_mode else vision.RunningMode.IMAGE
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=running_mode,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._video_mode = video_mode
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def __enter__(self) -> HandLandmarkExtractor:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Release the native landmarker."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def detect(self, image_bgr: np.ndarray, timestamp_ms: int | None = None) -> HandDetection | None:
        """Detect the most confident hand in a BGR image.

        Args:
            image_bgr: OpenCV BGR image, as captured (not mirrored).
            timestamp_ms: Monotonic frame timestamp; required in video mode.

        Returns:
            The detection, or None when no hand is found.
        """
        import cv2
        import mediapipe as mp

        if image_bgr is None or image_bgr.ndim != 3:
            raise ValueError("Expected a BGR image of shape (H, W, 3)")
        height, width = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        if self._video_mode:
            if timestamp_ms is None:
                raise ValueError("timestamp_ms is required in video mode")
            result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        else:
            result = self._landmarker.detect(mp_image)
        if not result.hand_landmarks:
            return None
        best = int(np.argmax([h[0].score for h in result.handedness]))
        raw = np.array([[lm.x, lm.y, lm.z] for lm in result.hand_landmarks[best]], dtype=np.float64)
        category = result.handedness[best][0]
        return HandDetection(
            points_px=to_pixel_space(raw, width, height),
            handedness=category.category_name,
            handedness_score=float(category.score),
        )
