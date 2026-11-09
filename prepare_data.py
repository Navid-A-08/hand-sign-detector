"""Download the datasets and extract normalized hand landmarks to parquet.

Usage:
    python prepare_data.py               # download everything and extract
    python prepare_data.py --model-only  # only fetch hand_landmarker.task
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from hand_sign_detector.config import load_config  # noqa: E402
from hand_sign_detector.datasets import (  # noqa: E402
    SOURCE_LICENSES,
    danrasband_items,
    download_sources,
    extract_images,
    file_md5,
    load_aslnow,
    load_borisgraudt,
    near_duplicate_mask,
    sau_items,
)
from hand_sign_detector.logging_utils import setup_logging  # noqa: E402

logger = logging.getLogger("prepare_data")


def ensure_landmarker_model(path: Path, url: str) -> None:
    """Download the MediaPipe hand landmarker model if it is missing."""
    if path.is_file():
        logger.info("Landmarker model present at %s", path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s", url)
    urllib.request.urlretrieve(url, path)


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--model-only", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="Max images per image source (0 = all); for quick tests")
    args = parser.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    ensure_landmarker_model(cfg.paths.landmarker_model, cfg.landmarker.model_url)
    if args.model_only:
        return

    out_dir = cfg.paths.landmarks_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    letters, nothing = cfg.classes.letters, cfg.classes.nothing_label
    roots = download_sources()
    summary: dict = {"roots": {k: str(v) for k, v in roots.items()}, "sources": {}}

    held_out_md5: set[str] = set()
    held_out_paths: list[str] = []
    if "danrasband" in roots:
        items = danrasband_items(roots["danrasband"], letters, nothing)
        held_out_md5 = {file_md5(item.path) for item in items}
        held_out_paths = [item.path for item in items]
        if args.limit:
            items = items[:: max(1, len(items) // args.limit)]
        frame = extract_images(items, cfg.paths.landmarker_model, args.workers)
        frame.to_parquet(out_dir / "danrasband.parquet", index=False)
        logger.info("danrasband: %d images, %d with a hand", len(frame), int(frame["detected"].sum()))

    if "sau" in roots:
        items, drops = sau_items(roots["sau"], letters, nothing, held_out_md5)
        near_dup = near_duplicate_mask([item.path for item in items], held_out_paths, workers=args.workers)
        drops["held_out_near_duplicate"] = int(near_dup.sum())
        dropped_by_source: dict[str, int] = {}
        for item, is_dup in zip(items, near_dup):
            if is_dup:
                dropped_by_source[item.source] = dropped_by_source.get(item.source, 0) + 1
        drops["held_out_near_duplicate_by_source"] = dropped_by_source
        items = [item for item, is_dup in zip(items, near_dup) if not is_dup]
        summary["sau_drops"] = drops
        logger.info("Sau: %d images kept, drops=%s", len(items), drops)
        if args.limit:
            items = items[:: max(1, len(items) // args.limit)]
        frame = extract_images(items, cfg.paths.landmarker_model, args.workers)
        for source, part in frame.groupby("source"):
            part.to_parquet(out_dir / f"{source}.parquet", index=False)
            logger.info("%s: %d images, %d with a hand", source, len(part), int(part["detected"].sum()))

    if "borisgraudt" in roots:
        load_borisgraudt(roots["borisgraudt"], letters, nothing).to_parquet(out_dir / "borisgraudt.parquet", index=False)
    if "aslnow" in roots:
        load_aslnow(roots["aslnow"], letters, nothing).to_parquet(out_dir / "aslnow.parquet", index=False)

    import pandas as pd

    for path in sorted(out_dir.glob("*.parquet")):
        frame = pd.read_parquet(path, columns=["source", "label", "detected"])
        source = path.stem
        summary["sources"][source] = {
            "rows": int(len(frame)),
            "detected": int(frame["detected"].sum()),
            "license": SOURCE_LICENSES.get(source, "unknown"),
            "per_label": frame.groupby("label").size().to_dict(),
        }
    report_path = cfg.paths.reports_dir / "data_summary.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Wrote %s", report_path)


if __name__ == "__main__":
    main()
