"""Evaluate saved models on the held-out danrasband test set.

The danrasband images are never used for training. They are also present
byte-for-byte inside the Sau dataset, and ``prepare_data.py`` drops those copies.

Usage:
    python evaluate.py                 # clean models
    python evaluate.py --tag with_gpl  # ablation models
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import pandas as pd  # noqa: E402

from hand_sign_detector.config import load_config  # noqa: E402
from hand_sign_detector.evaluation import evaluate_frame, filter_labels, save_confusion, summarize, write_json  # noqa: E402
from hand_sign_detector.logging_utils import setup_logging  # noqa: E402
from hand_sign_detector.models import load_model  # noqa: E402

logger = logging.getLogger("evaluate")


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--tag", default="clean")
    parser.add_argument("--models", nargs="+", default=["mlp", "random_forest"])
    args = parser.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    test_path = cfg.paths.landmarks_dir / "danrasband.parquet"
    frame = filter_labels(pd.read_parquet(test_path), cfg.classes.labels)
    reports_dir = cfg.paths.reports_dir / args.tag
    results: dict = {"test_set": "danrasband (held out)", "n_images": int(len(frame))}
    for name in args.models:
        path = cfg.paths.models_dir / f"{name}_{args.tag}.joblib"
        if not path.is_file():
            logger.warning("Missing %s; run train.py first", path)
            continue
        model, _ = load_model(path)
        metrics = evaluate_frame(model, frame, cfg.classes.labels, cfg.classes.nothing_label)
        save_confusion(metrics, reports_dir / "confusion" / f"danrasband_{name}", f"danrasband held-out [{name}, {args.tag}]")
        results[name] = summarize(metrics)
        results[name]["per_class_accuracy"] = metrics["per_class_accuracy"]
        results[name]["top_confused_pairs"] = metrics["top_confused_pairs"]
        logger.info(
            "%s: detection(letters)=%.3f cls_acc=%.3f end_to_end=%.3f",
            name,
            metrics["detection_rate_letters"],
            metrics["classification_accuracy"],
            metrics["end_to_end_accuracy"],
        )
    write_json(results, reports_dir / "danrasband_metrics.json")
    logger.info("Wrote %s", reports_dir / "danrasband_metrics.json")


if __name__ == "__main__":
    main()
