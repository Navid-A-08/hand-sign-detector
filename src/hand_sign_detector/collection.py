"""Camera-free part of webcam data collection: buffering labelled landmark rows."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .datasets import FEATURE_COLUMNS, META_COLUMNS
from .landmarks import HandDetection

_SIGNER_ID = re.compile(r"^[A-Za-z0-9_-]{1,32}$")


def validate_signer_id(signer_id: str) -> str:
    """Return the signer ID if it is a short, file-name-safe token."""
    if not _SIGNER_ID.match(signer_id):
        raise ValueError("signer_id must be 1-32 characters of letters, digits, '_' or '-'")
    return signer_id


class CollectionSession:
    """Accumulate labelled webcam samples for one signer and one session."""

    def __init__(self, signer_id: str, labels: list[str], session_id: str | None = None) -> None:
        self.signer_id = validate_signer_id(signer_id)
        self.labels = labels
        self.session_id = session_id or datetime.now().strftime("%Y%m%d-%H%M%S")
        self._rows: list[dict] = []

    def add(self, label: str, detection: HandDetection | None, frame_index: int) -> bool:
        """Record one frame. Frames without a usable hand are recorded as not detected.

        Returns:
            True if a hand was detected and its features were stored.
        """
        if label not in self.labels:
            raise ValueError(f"Unknown label '{label}'")
        row = {
            "source": "webcam",
            "group": f"webcam:{self.signer_id}:{self.session_id}",
            "signer_id": self.signer_id,
            "label": label,
            "detected": False,
            "handedness": None,
            "handedness_score": np.nan,
            "path": f"frame:{frame_index}",
        }
        features = np.full(len(FEATURE_COLUMNS), np.nan, dtype=np.float32)
        if detection is not None:
            try:
                features = detection.features
                row.update(detected=True, handedness=detection.handedness, handedness_score=detection.handedness_score)
            except ValueError:
                pass
        row.update(zip(FEATURE_COLUMNS, features.tolist()))
        self._rows.append(row)
        return bool(row["detected"])

    def counts(self) -> dict[str, int]:
        """Detected samples per label so far."""
        out: dict[str, int] = {}
        for row in self._rows:
            if row["detected"]:
                out[row["label"]] = out.get(row["label"], 0) + 1
        return out

    def to_frame(self) -> pd.DataFrame:
        """All rows as a landmark table (same schema as the dataset tables)."""
        frame = pd.DataFrame(self._rows, columns=[*META_COLUMNS, *FEATURE_COLUMNS])
        frame[FEATURE_COLUMNS] = frame[FEATURE_COLUMNS].astype(np.float32)
        return frame

    def save(self, out_dir: Path) -> Path:
        """Write the session to ``<out_dir>/webcam_<signer>_<session>.parquet``."""
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"webcam_{self.signer_id}_{self.session_id}.parquet"
        self.to_frame().to_parquet(path, index=False)
        return path

    def __len__(self) -> int:
        return len(self._rows)
