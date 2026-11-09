"""Record your own labelled webcam landmarks, tagged with a signer ID.

Saved tables use the same schema as the dataset tables, so ``train.py`` picks
them up from ``data/landmarks`` automatically (source = "webcam").

Usage:
    python collect.py --signer navid
Keys:
    a-y   choose the letter to record (j and z are ignored in v1)
    0     choose 'nothing' (hand visible but not signing)
    space start / stop recording
    Esc   save and quit
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import cv2  # noqa: E402

from hand_sign_detector.collection import CollectionSession  # noqa: E402
from hand_sign_detector.config import load_config  # noqa: E402
from hand_sign_detector.landmarks import HandLandmarkExtractor  # noqa: E402
from hand_sign_detector.logging_utils import setup_logging  # noqa: E402
from hand_sign_detector.predictor import draw_hand  # noqa: E402

logger = logging.getLogger("collect")


def key_to_label(key: int, letters: list[str], nothing_label: str) -> str | None:
    """Map a pressed key to a class label, or None if it is not a label key."""
    if key == ord("0"):
        return nothing_label
    if 0 <= key < 256:
        char = chr(key).upper()
        if char in letters:
            return char
    return None


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--signer", required=True, help="Signer ID, e.g. 'navid'")
    parser.add_argument("--config", default=None)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--every", type=int, default=2, help="Keep every Nth frame while recording")
    args = parser.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    session = CollectionSession(args.signer, cfg.classes.labels)
    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}")
    label = cfg.classes.letters[0]
    recording = False
    frame_index = 0
    start = time.monotonic()
    with HandLandmarkExtractor(cfg.paths.landmarker_model, video_mode=True) as extractor:
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frame_index += 1
                detection = extractor.detect(frame, int((time.monotonic() - start) * 1000))
                if recording and frame_index % args.every == 0:
                    session.add(label, detection, frame_index)
                if detection is not None:
                    draw_hand(frame, detection)
                display = cv2.flip(frame, 1)
                status = f"{'REC' if recording else 'paused'} | label {label} | {session.counts().get(label, 0)} samples"
                cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255) if recording else (255, 255, 255), 2)
                cv2.imshow("collect", display)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break
                if key == ord(" "):
                    recording = not recording
                elif (new_label := key_to_label(key, cfg.classes.letters, cfg.classes.nothing_label)) is not None:
                    label = new_label
                    recording = False
        finally:
            capture.release()
            cv2.destroyAllWindows()
    if len(session):
        path = session.save(cfg.paths.landmarks_dir)
        logger.info("Saved %d frames (%s) to %s", len(session), session.counts(), path)


if __name__ == "__main__":
    main()
