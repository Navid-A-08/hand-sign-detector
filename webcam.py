"""Live ASL letter recognition from a webcam.

Shows the smoothed label (majority vote over recent frames), 'Uncertain' when
the top probability is below the threshold, and the top-2 candidates.
Frames go to MediaPipe as captured; the window shows a mirrored (selfie) view.

Usage:
    python webcam.py [--camera 0] [--model models/mlp_clean.joblib]
Keys: q / Esc to quit.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from hand_sign_detector.config import load_config  # noqa: E402
from hand_sign_detector.inference import MajorityVoteSmoother, Prediction  # noqa: E402
from hand_sign_detector.landmarks import HandLandmarkExtractor  # noqa: E402
from hand_sign_detector.logging_utils import setup_logging  # noqa: E402
from hand_sign_detector.predictor import SignPredictor, draw_hand  # noqa: E402

logger = logging.getLogger("webcam")


def format_overlay(smoothed: str, prediction: Prediction, fps: float) -> list[str]:
    """Build the overlay text lines (kept separate so it can be unit tested)."""
    lines = [f"Sign: {smoothed}"]
    lines += [f"{rank}. {label} {prob:.0%}" for rank, (label, prob) in enumerate(prediction.top_k, start=1)]
    lines.append(f"{fps:.0f} FPS")
    return lines


def draw_overlay(frame: np.ndarray, lines: list[str]) -> None:
    """Draw text lines in the top-left corner."""
    for i, line in enumerate(lines):
        scale = 1.0 if i == 0 else 0.6
        y = 35 + i * 28
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2)


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", type=Path, default=None)
    args = parser.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    model_path = args.model or cfg.paths.models_dir / "mlp_clean.joblib"
    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}")
    smoother = MajorityVoteSmoother(cfg.inference.smoothing_window)
    start = time.monotonic()
    last = start
    fps = 0.0
    with HandLandmarkExtractor(cfg.paths.landmarker_model, video_mode=True) as extractor:
        predictor = SignPredictor(model_path, extractor, cfg.inference.confidence_threshold, cfg.inference.top_k)
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    logger.warning("Camera returned no frame")
                    break
                now = time.monotonic()
                prediction, detection = predictor.predict_image(frame, int((now - start) * 1000))
                smoothed = smoother.update(prediction.label)
                fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)
                last = now
                if detection is not None:
                    draw_hand(frame, detection)
                display = cv2.flip(frame, 1)
                draw_overlay(display, format_overlay(smoothed, prediction, fps))
                cv2.imshow("hand-sign-detector", display)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
        finally:
            capture.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
