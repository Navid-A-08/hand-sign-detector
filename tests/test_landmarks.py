"""Normalization invariances and mirroring conventions."""

from __future__ import annotations

import numpy as np
import pytest

from hand_sign_detector.landmarks import (
    FEATURE_DIM,
    MIDDLE_MCP,
    WRIST,
    mirror_x,
    normalize_landmarks,
    palm_size,
    to_pixel_space,
)


def test_output_shape_and_dtype(hand_points: np.ndarray) -> None:
    features = normalize_landmarks(hand_points, "Right")
    assert features.shape == (FEATURE_DIM,)
    assert features.dtype == np.float32


def test_wrist_at_origin_and_unit_palm(hand_points: np.ndarray) -> None:
    pts = normalize_landmarks(hand_points, "Right").reshape(21, 3)
    np.testing.assert_allclose(pts[WRIST], 0.0, atol=1e-6)
    assert np.linalg.norm(pts[MIDDLE_MCP]) == pytest.approx(1.0, abs=1e-5)


def test_translation_invariance(hand_points: np.ndarray) -> None:
    shifted = hand_points + np.array([37.0, -12.0, 4.0])
    np.testing.assert_allclose(normalize_landmarks(shifted), normalize_landmarks(hand_points), atol=1e-5)


@pytest.mark.parametrize("scale", [0.1, 0.5, 3.0, 20.0])
def test_scale_invariance(hand_points: np.ndarray, scale: float) -> None:
    np.testing.assert_allclose(normalize_landmarks(hand_points * scale), normalize_landmarks(hand_points), atol=1e-5)


def test_not_rotation_invariant(hand_points: np.ndarray) -> None:
    """Orientation must survive normalization (H vs U, G vs Q, K vs P differ by it)."""
    theta = np.deg2rad(90.0)
    rot = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    rotated = (hand_points - hand_points[0]) @ rot.T + hand_points[0]
    assert not np.allclose(normalize_landmarks(rotated), normalize_landmarks(hand_points), atol=1e-3)


def test_left_hand_is_mirrored_to_canonical_right(hand_points: np.ndarray) -> None:
    """A mirrored image of a right hand, labelled Left, must map to the same features."""
    width = 640.0
    mirrored_image = hand_points.copy()
    mirrored_image[:, 0] = width - mirrored_image[:, 0]
    np.testing.assert_allclose(
        normalize_landmarks(mirrored_image, "Left"),
        normalize_landmarks(hand_points, "Right"),
        atol=1e-5,
    )


def test_unknown_handedness_is_not_mirrored(hand_points: np.ndarray) -> None:
    np.testing.assert_allclose(normalize_landmarks(hand_points, None), normalize_landmarks(hand_points, "Right"))
    assert not np.allclose(normalize_landmarks(hand_points, "Left"), normalize_landmarks(hand_points, "Right"))


def test_mirror_is_involution(hand_points: np.ndarray) -> None:
    np.testing.assert_allclose(mirror_x(mirror_x(hand_points)), hand_points)


def test_pixel_space_keeps_aspect_ratio() -> None:
    """A 45-degree segment in a 2:1 frame stays 45 degrees after conversion."""
    points = np.zeros((21, 3))
    points[9] = [0.25, 0.5, 0.0]
    px = to_pixel_space(points, width=200.0, height=100.0)
    assert px[9, 0] == pytest.approx(50.0)
    assert px[9, 1] == pytest.approx(50.0)


def test_accepts_flat_vector(hand_points: np.ndarray) -> None:
    np.testing.assert_allclose(normalize_landmarks(hand_points.ravel()), normalize_landmarks(hand_points))


def test_degenerate_palm_raises() -> None:
    with pytest.raises(ValueError):
        normalize_landmarks(np.zeros((21, 3)))


def test_wrong_shape_and_nan_raise(hand_points: np.ndarray) -> None:
    with pytest.raises(ValueError):
        normalize_landmarks(np.zeros(60))
    bad = hand_points.copy()
    bad[3, 1] = np.nan
    with pytest.raises(ValueError):
        normalize_landmarks(bad)


def test_palm_size(hand_points: np.ndarray) -> None:
    assert palm_size(hand_points) == pytest.approx(np.linalg.norm(hand_points[9] - hand_points[0]))
