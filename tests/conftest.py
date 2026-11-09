"""Shared pytest fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def hand_points() -> np.ndarray:
    """A deterministic, non-degenerate (21, 3) hand in pixel-aspect units."""
    rng = np.random.default_rng(0)
    points = rng.uniform(50.0, 150.0, size=(21, 3))
    points[0] = [100.0, 180.0, 0.0]
    points[9] = [100.0, 100.0, -5.0]
    return points
