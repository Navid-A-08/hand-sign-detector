"""Predict the ASL letter in one or more images.

Usage:
    python predict_image.py path/to/img.jpg [more.jpg ...] [--model models/mlp_clean.joblib] [--annotate out_dir]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import cv2  # noqa: E402

from hand_sign_detector.config import load_config  # noqa: E402
from hand_sign_detector.landmarks import HandLandmarkExtractor  # noqa: E402
from hand_sign_detector.logging_utils import setup_logging  # noqa: E402
from hand_sign_detector.predictor import SignPredictor, draw_hand  # noqa: E402

logger = logging.getLogger("predict_image")


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--config", default=None)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--annotate", type=Path, default=None, help="Write annotated copies to this folder")
    args = parser.parse_args()
    setup_logging(logging.WARNING)
    cfg = load_config(args.config)
    model_path = args.model or cfg.paths.models_dir / "mlp_clean.joblib"
    with HandLandmarkExtractor(cfg.paths.landmarker_model) as extractor:
        predictor = SignPredictor(model_path, extractor, cfg.inference.confidence_threshold, cfg.inference.top_k)
        for path in args.images:
            image = cv2.imread(str(path))
            if image is None:
                logger.error("Could not read %s", path)
                continue
            prediction, detection = predictor.predict_image(image)
            record = {
                "image": str(path),
                "hand_detected": detection is not None,
                "handedness": detection.handedness if detection else None,
                "label": prediction.label,
                "confidence": round(prediction.confidence, 4),
                "top_k": [[label, round(p, 4)] for label, p in prediction.top_k],
            }
            print(json.dumps(record))
            if args.annotate:
                args.annotate.mkdir(parents=True, exist_ok=True)
                if detection is not None:
                    draw_hand(image, detection)
                text = " | ".join(f"{label} {p:.2f}" for label, p in prediction.top_k)
                cv2.putText(image, f"{prediction.label}: {text}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                cv2.imwrite(str(args.annotate / path.name), image)


if __name__ == "__main__":
    main()
