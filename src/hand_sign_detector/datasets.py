"""Dataset download, landmark extraction and loading.

Every row in a landmark table is one sample with these columns:
``source, group, signer_id, label, detected, handedness, handedness_score, path, f0..f62``.
Rows with ``detected == False`` keep NaN features so detection rates can be
reported separately from classification accuracy.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .landmarks import FEATURE_DIM, HandLandmarkExtractor, normalize_landmarks, to_pixel_space

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [f"f{i}" for i in range(FEATURE_DIM)]
META_COLUMNS = ["source", "group", "signer_id", "label", "detected", "handedness", "handedness_score", "path"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}

KAGGLE_DANRASBAND = "danrasband/asl-alphabet-test"
KAGGLE_SAU = "debashishsau/aslamerican-sign-language-aplhabet-dataset"
KAGGLE_BORIS = "borisgraudt/asl-alphabet-hand-landmarks"
HF_ASLNOW = "sid220/asl-now-fingerspelling"

# License of each source as stated on its dataset page.
SOURCE_LICENSES: dict[str, str] = {
    "danrasband": "CC0-1.0",
    "sau_paren": "CC0-1.0 (Sau compilation)",
    "sau_rotate": "CC0-1.0 (Sau compilation)",
    "sau_numeric": "CC0-1.0 (Sau compilation)",
    "grassknoted": "GPL-2.0 (grassknoted ASL Alphabet, re-hosted inside Sau)",
    "borisgraudt": "CC0-1.0",
    "aslnow": "MIT",
    "webcam": "own recordings",
}

_SAU_ROTATE = re.compile(r"^([a-z]+)_(\d+)_rotate_\d+\.jpe?g$", re.IGNORECASE)
_SAU_PAREN = re.compile(r"^[A-Za-z]+ \(\d+\)\.jpe?g$")
_GRASSKNOTED = re.compile(r"^[A-Za-z]+\d+\.jpe?g$")
_DANRASBAND = re.compile(r"^[A-Za-z]+\d+_test\.jpe?g$")
_NUMERIC = re.compile(r"^\d+\.jpe?g$")


@dataclass(frozen=True)
class ImageItem:
    """One image to run the landmarker on."""

    path: str
    source: str
    group: str
    label: str


def map_label(folder: str, letters: list[str], nothing_label: str) -> str | None:
    """Map a dataset folder name to a class label, or None to skip it (J, Z, del, space)."""
    name = folder.strip()
    if name.lower() == nothing_label.lower():
        return nothing_label
    upper = name.upper()
    return upper if upper in letters else None


def classify_sau_file(filename: str) -> tuple[str, str] | None:
    """Assign a file inside the Sau compilation to its sub-source and group.

    The Sau dataset pools several datasets. File-name patterns identify them:
    ``A123.jpg`` is the grassknoted ASL Alphabet set, ``A0001_test.jpg`` is the
    danrasband test set (returned as ``danrasband`` so callers can drop it),
    ``A (12).jpg`` and ``12.jpg`` are other pooled sets, and
    ``a_12_rotate_3.jpeg`` are rotated copies grouped by their base image.

    Returns:
        ``(source, group)`` or None if the name matches no known pattern.
    """
    if _DANRASBAND.match(filename):
        return "danrasband", "danrasband"
    if match := _SAU_ROTATE.match(filename):
        return "sau_rotate", f"sau_rotate:{match.group(1).lower()}_{match.group(2)}"
    if _SAU_PAREN.match(filename):
        return "sau_paren", "sau_paren"
    if _GRASSKNOTED.match(filename):
        return "grassknoted", "grassknoted"
    if _NUMERIC.match(filename):
        return "sau_numeric", "sau_numeric"
    return None


def file_md5(path: Path | str) -> str:
    """Return the MD5 hex digest of a file."""
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_sources() -> dict[str, Path]:
    """Download every source that is reachable without credentials.

    Returns:
        Mapping of source name to local root directory. Failed sources are logged and omitted.
    """
    roots: dict[str, Path] = {}
    try:
        import kagglehub

        for key, handle in [("danrasband", KAGGLE_DANRASBAND), ("sau", KAGGLE_SAU), ("borisgraudt", KAGGLE_BORIS)]:
            try:
                roots[key] = Path(kagglehub.dataset_download(handle))
                logger.info("Downloaded %s -> %s", handle, roots[key])
            except Exception as exc:  # noqa: BLE001 - report and continue with what is reachable
                logger.error("Could not download %s: %s", handle, exc)
    except ImportError:
        logger.error("kagglehub is not installed; skipping Kaggle sources")
    try:
        from huggingface_hub import snapshot_download

        roots["aslnow"] = Path(snapshot_download(HF_ASLNOW, repo_type="dataset"))
        logger.info("Downloaded %s -> %s", HF_ASLNOW, roots["aslnow"])
    except Exception as exc:  # noqa: BLE001
        logger.error("Could not download %s: %s", HF_ASLNOW, exc)
    return roots


def _iter_class_images(root: Path) -> list[tuple[str, Path]]:
    """List ``(folder_name, image_path)`` for a class-per-folder image tree."""
    items: list[tuple[str, Path]] = []
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for path in sorted(class_dir.iterdir()):
            if path.suffix.lower() in IMAGE_SUFFIXES:
                items.append((class_dir.name, path))
    return items


def danrasband_items(root: Path, letters: list[str], nothing_label: str) -> list[ImageItem]:
    """List the danrasband images (top-level class folders only; the nested copy is identical)."""
    items: list[ImageItem] = []
    for folder, path in _iter_class_images(root):
        if folder == "asl-alphabet-test":
            continue
        label = map_label(folder, letters, nothing_label)
        if label is not None:
            items.append(ImageItem(str(path), "danrasband", f"danrasband:{path.stem}", label))
    return items


def sau_items(
    root: Path,
    letters: list[str],
    nothing_label: str,
    exclude_md5: set[str],
) -> tuple[list[ImageItem], dict[str, int]]:
    """List Sau training images, dropping held-out test copies and exact duplicates.

    Byte-identical copies are attributed conservatively: any image whose bytes
    match a grassknoted-pattern file (``A123.jpg``) is treated as grassknoted
    (GPL-2), even if it also appears under another name.

    Args:
        root: Sau dataset root (the directory that contains ``ASL_Alphabet_Dataset``).
        letters: Letter labels to keep.
        nothing_label: Label of the 'nothing' class.
        exclude_md5: Hashes of held-out images (danrasband) that must not be trained on.

    Returns:
        The image items and a dict of drop counters for the report.
    """
    train_root = root / "ASL_Alphabet_Dataset" / "asl_alphabet_train"
    counts = {
        "held_out_by_name": 0,
        "held_out_by_hash": 0,
        "exact_duplicate": 0,
        "reattributed_to_grassknoted": 0,
        "unknown_pattern": 0,
        "skipped_label": 0,
    }
    candidates: list[tuple[str, Path, str, str, str]] = []
    grassknoted_md5: set[str] = set()
    for folder, path in _iter_class_images(train_root):
        label = map_label(folder, letters, nothing_label)
        if label is None:
            counts["skipped_label"] += 1
            continue
        assigned = classify_sau_file(path.name)
        if assigned is None:
            counts["unknown_pattern"] += 1
            continue
        source, group = assigned
        if source == "danrasband":
            counts["held_out_by_name"] += 1
            continue
        digest = file_md5(path)
        if source == "grassknoted":
            grassknoted_md5.add(digest)
        candidates.append((label, path, source, group, digest))

    seen: set[str] = set()
    items: list[ImageItem] = []
    for label, path, source, group, digest in candidates:
        if digest in exclude_md5:
            counts["held_out_by_hash"] += 1
            continue
        if digest in grassknoted_md5 and source != "grassknoted":
            counts["reattributed_to_grassknoted"] += 1
            continue
        if digest in seen:
            counts["exact_duplicate"] += 1
            continue
        seen.add(digest)
        items.append(ImageItem(str(path), source, group, label))
    return items, counts


THUMBNAIL_SIZE = 32
# 1 - Pearson correlation of 32x32 grayscale thumbnails. Re-encoded / resized
# copies of danrasband images inside Sau score < 0.01; the next-closest
# unrelated images score > 0.1 (measured 2026-11-09, see README).
NEAR_DUPLICATE_THRESHOLD = 0.1


def image_thumbnail(path: str) -> np.ndarray:
    """Return a zero-mean, unit-norm 32x32 grayscale thumbnail (flattened)."""
    import cv2

    image = cv2.imread(path, cv2.IMREAD_REDUCED_GRAYSCALE_4)
    if image is None:
        return np.zeros(THUMBNAIL_SIZE * THUMBNAIL_SIZE, dtype=np.float32)
    vec = cv2.resize(image, (THUMBNAIL_SIZE, THUMBNAIL_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    vec -= vec.mean()
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


def near_duplicate_mask(
    paths: list[str],
    reference_paths: list[str],
    threshold: float = NEAR_DUPLICATE_THRESHOLD,
    workers: int = 4,
) -> np.ndarray:
    """Flag images that are visually near-identical to any reference image.

    Catches resized or re-encoded copies that an exact hash misses.

    Returns:
        Boolean array, True where ``1 - corr`` to the nearest reference is below ``threshold``.
    """
    from sklearn.neighbors import NearestNeighbors

    if not paths or not reference_paths:
        return np.zeros(len(paths), dtype=bool)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        reference = np.stack(list(pool.map(image_thumbnail, reference_paths, chunksize=32)))
        candidates = np.stack(list(pool.map(image_thumbnail, paths, chunksize=64)))
    distance, _ = NearestNeighbors(n_neighbors=1).fit(reference).kneighbors(candidates)
    # For unit vectors, squared euclidean distance / 2 == 1 - correlation.
    return (distance.ravel() ** 2 / 2.0) < threshold


_WORKER_EXTRACTOR: HandLandmarkExtractor | None = None


def _init_worker(model_path: str) -> None:
    global _WORKER_EXTRACTOR
    os.environ.setdefault("GLOG_minloglevel", "2")
    _WORKER_EXTRACTOR = HandLandmarkExtractor(model_path)


def _extract_one(item: ImageItem) -> dict:
    import cv2

    row: dict = {
        "source": item.source,
        "group": item.group,
        "signer_id": "unknown",
        "label": item.label,
        "detected": False,
        "handedness": None,
        "handedness_score": np.nan,
        "path": item.path,
    }
    features = np.full(FEATURE_DIM, np.nan, dtype=np.float32)
    image = cv2.imread(item.path)
    if image is not None and _WORKER_EXTRACTOR is not None:
        detection = _WORKER_EXTRACTOR.detect(image)
        if detection is not None:
            try:
                features = detection.features
                row.update(detected=True, handedness=detection.handedness, handedness_score=detection.handedness_score)
            except ValueError:
                pass
    row.update(zip(FEATURE_COLUMNS, features.tolist()))
    return row


def extract_images(items: list[ImageItem], model_path: Path, workers: int = 4) -> pd.DataFrame:
    """Run the landmarker over images in parallel and return a landmark table."""
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(str(model_path),)) as pool:
        for index, row in enumerate(pool.map(_extract_one, items, chunksize=64), start=1):
            rows.append(row)
            if index % 5000 == 0:
                logger.info("Extracted %d / %d images", index, len(items))
    return _to_frame(rows)


def _to_frame(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=[*META_COLUMNS, *FEATURE_COLUMNS])
    frame[FEATURE_COLUMNS] = frame[FEATURE_COLUMNS].astype(np.float32)
    return frame


def _landmark_row(points: np.ndarray, source: str, group: str, label: str, path: str) -> dict:
    """Build a row from a pre-extracted landmark array with unknown handedness and aspect."""
    row = {
        "source": source,
        "group": group,
        "signer_id": "unknown",
        "label": label,
        "detected": True,
        "handedness": None,
        "handedness_score": np.nan,
        "path": path,
    }
    features = normalize_landmarks(to_pixel_space(points, 1.0, 1.0), handedness=None)
    row.update(zip(FEATURE_COLUMNS, features.tolist()))
    return row


def load_borisgraudt(root: Path, letters: list[str], nothing_label: str) -> pd.DataFrame:
    """Load the borisgraudt ``.npy`` landmark set (no handedness or image size recorded)."""
    rows: list[dict] = []
    for class_dir in sorted(p for p in (root / "landmarks").iterdir() if p.is_dir()):
        label = map_label(class_dir.name, letters, nothing_label)
        if label is None:
            continue
        for path in sorted(class_dir.glob("*.npy")):
            points = np.load(path)
            try:
                rows.append(_landmark_row(points, "borisgraudt", "borisgraudt", label, str(path)))
            except ValueError:
                logger.warning("Skipping degenerate sample %s", path)
    return _to_frame(rows)


def load_aslnow(root: Path, letters: list[str], nothing_label: str) -> pd.DataFrame:
    """Load the ASLNow! JSON landmark set (no signer IDs, handedness or image size recorded)."""
    rows: list[dict] = []
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        label = map_label(class_dir.name, letters, nothing_label)
        if label is None:
            continue
        for path in sorted(class_dir.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            points = np.array([[p["x"], p["y"], p["z"]] for p in data], dtype=np.float64)
            if points.shape != (21, 3):
                logger.warning("Skipping %s with shape %s", path, points.shape)
                continue
            try:
                rows.append(_landmark_row(points, "aslnow", "aslnow", label, str(path)))
            except ValueError:
                logger.warning("Skipping degenerate sample %s", path)
    return _to_frame(rows)


def load_landmark_tables(landmarks_dir: Path) -> pd.DataFrame:
    """Concatenate every ``*.parquet`` landmark table in a directory."""
    files = sorted(Path(landmarks_dir).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No landmark tables in {landmarks_dir}. Run prepare_data.py first.")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
