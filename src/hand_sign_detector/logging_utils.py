"""Logging setup shared by the command-line scripts."""

from __future__ import annotations

import logging
import os


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging and quiet MediaPipe/TFLite native logs."""
    os.environ.setdefault("GLOG_minloglevel", "2")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpx2", "httpcore", "urllib3", "huggingface_hub", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
