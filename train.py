"""Train the MLP and the baseline on extracted landmarks.

Runs a source-disjoint (leave-one-source-out) cross-validation, a random-split
reference (to show how much frame-burst leakage inflates scores), and then fits
the final models on all training sources. The danrasband set is never used
here; ``evaluate.py`` scores it as the external held-out test.

Usage:
    python train.py                    # default, GPL grassknoted subset excluded
    python train.py --include-gpl      # ablation that also trains on grassknoted
    python train.py --skip-cv          # only fit the final models
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from hand_sign_detector.config import AppConfig, load_config  # noqa: E402
from hand_sign_detector.datasets import load_landmark_tables  # noqa: E402
from hand_sign_detector.evaluation import (  # noqa: E402
    detected_xy,
    evaluate_frame,
    filter_labels,
    save_confusion,
    summarize,
    write_json,
)
from hand_sign_detector.logging_utils import setup_logging  # noqa: E402
from hand_sign_detector.models import build_model, save_model  # noqa: E402

logger = logging.getLogger("train")
HELD_OUT_SOURCES = {"danrasband"}
MIN_SAMPLES_PER_LABEL = 20


def model_specs(cfg: AppConfig) -> dict[str, dict]:
    """Return ``{model_name: params}`` for the MLP and the configured baseline."""
    baseline = dict(cfg.training.baseline)
    kind = baseline.pop("kind", "random_forest")
    return {"mlp": dict(cfg.training.mlp), kind: baseline}


def load_training_frame(cfg: AppConfig, include_gpl: bool, extra_exclude: list[str] | None = None) -> pd.DataFrame:
    """Load landmark tables, drop held-out and excluded sources, keep known labels."""
    frame = load_landmark_tables(cfg.paths.landmarks_dir)
    excluded = set(HELD_OUT_SOURCES) | set(extra_exclude or [])
    if not include_gpl:
        excluded |= set(cfg.training.exclude_sources)
    frame = frame[~frame["source"].isin(excluded)]
    frame = filter_labels(frame, cfg.classes.labels)
    detected_counts = frame.loc[frame["detected"], "label"].value_counts()
    too_few = sorted(label for label in cfg.classes.labels if detected_counts.get(label, 0) < MIN_SAMPLES_PER_LABEL)
    if too_few:
        logger.warning("Dropping labels with < %d detected samples: %s", MIN_SAMPLES_PER_LABEL, too_few)
        frame = frame[~frame["label"].isin(too_few)]
    cap = cfg.training.max_per_group_label
    if cap > 0:
        frame = frame.groupby(["source", "group", "label"], group_keys=False).apply(
            lambda g: g.sample(min(len(g), cap), random_state=cfg.training.seed)
        )
    return frame.reset_index(drop=True)


def fit(name: str, params: dict, frame: pd.DataFrame, seed: int):
    """Fit one model on the detected rows of ``frame``."""
    x, y = detected_xy(frame)
    model = build_model(name, params, seed)
    start = time.perf_counter()
    model.fit(x, y)
    logger.info("Fitted %s on %d samples in %.1fs", name, len(y), time.perf_counter() - start)
    return model


def run_cv(cfg: AppConfig, frame: pd.DataFrame, reports_dir: Path) -> dict:
    """Leave-one-source-out CV plus a random-split reference."""
    seed = cfg.training.seed
    labels = cfg.classes.labels
    nothing = cfg.classes.nothing_label
    results: dict = {"leave_one_source_out": {}, "random_split_reference": {}}
    sources = sorted(frame["source"].unique())
    for held_out in sources:
        train_part = frame[frame["source"] != held_out]
        test_part = frame[frame["source"] == held_out]
        missing = sorted(set(test_part["label"]) - set(train_part["label"]))
        fold: dict = {"n_train_detected": int(train_part["detected"].sum()), "labels_missing_from_train": missing}
        for name, params in model_specs(cfg).items():
            model = fit(name, params, train_part, seed)
            metrics = evaluate_frame(model, test_part, labels, nothing)
            save_confusion(metrics, reports_dir / "confusion" / f"loso_{held_out}_{name}", f"LOSO held-out={held_out} [{name}]")
            fold[name] = summarize(metrics)
            fold[name]["per_class_accuracy"] = metrics["per_class_accuracy"]
            logger.info(
                "LOSO %s %s: detection=%.3f cls_acc=%.3f",
                held_out,
                name,
                metrics["detection_rate_letters"] or float("nan"),
                metrics["classification_accuracy"] or float("nan"),
            )
        results["leave_one_source_out"][held_out] = fold

    detected = frame[frame["detected"]]
    train_part, test_part = train_test_split(detected, test_size=0.2, random_state=seed, stratify=detected["label"])
    for name, params in model_specs(cfg).items():
        model = fit(name, params, train_part, seed)
        metrics = evaluate_frame(model, test_part, labels, nothing)
        results["random_split_reference"][name] = summarize(metrics)
        logger.info("Random split %s: cls_acc=%.3f", name, metrics["classification_accuracy"])
    return results


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--include-gpl", action="store_true", help="Also train on the GPL-2 grassknoted subset")
    parser.add_argument("--skip-cv", action="store_true")
    parser.add_argument("--exclude", nargs="*", default=[], help="Extra sources to leave out (sensitivity checks)")
    parser.add_argument("--tag", default=None, help="Override the output tag (default: clean / with_gpl)")
    args = parser.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    tag = args.tag or ("with_gpl" if args.include_gpl else "clean")
    reports_dir = cfg.paths.reports_dir / tag
    frame = load_training_frame(cfg, args.include_gpl, args.exclude)
    counts = frame.groupby("source").agg(rows=("label", "size"), detected=("detected", "sum"))
    logger.info("Training sources:\n%s", counts)
    summary: dict = {
        "tag": tag,
        "sources": {s: {"rows": int(r.rows), "detected": int(r.detected)} for s, r in counts.iterrows()},
        "labels_present": sorted(frame.loc[frame["detected"], "label"].unique()),
    }
    if not args.skip_cv:
        summary["cv"] = run_cv(cfg, frame, reports_dir)
    for name, params in model_specs(cfg).items():
        model = fit(name, params, frame, cfg.training.seed)
        metadata = {
            "model": name,
            "tag": tag,
            "classes": [str(c) for c in model.classes_],
            "feature_dim": int(model.n_features_in_),
            "trained_on": summary["sources"],
        }
        save_model(model, cfg.paths.models_dir / f"{name}_{tag}.joblib", metadata)
    write_json(summary, reports_dir / "train_summary.json")
    logger.info("Wrote %s", reports_dir / "train_summary.json")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
