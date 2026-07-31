"""Tests for the backend-shared grasp predicate primitives."""

import numpy as np
import pytest

from so101_nexus.grasp import opposing_normals_ok

_UP = np.array([0.0, 0.0, 1.0])
_DOWN = np.array([0.0, 0.0, -1.0])


def test_pinch_is_accepted():
    """Anti-parallel inward normals are the definition of a pinch."""
    assert bool(opposing_normals_ok(_UP, _DOWN, threshold=0.3))


def test_same_side_straddle_is_rejected():
    """Both fingers pressing the same face is not a grasp, however hard they press."""
    assert not bool(opposing_normals_ok(_DOWN * 40.0, _DOWN * 40.0, threshold=0.3))


def test_magnitude_does_not_change_the_verdict():
    """The test is on direction only, so force-weighted sums need no normalization."""
    weak = opposing_normals_ok(_UP * 1e-3, _DOWN * 1e-3, threshold=0.3)
    strong = opposing_normals_ok(_UP * 1e3, _DOWN * 1e3, threshold=0.3)
    assert bool(weak) == bool(strong) is True


@pytest.mark.parametrize(
    "half_angle_deg,expected",
    # Fingers on the two flanks of a wedge, each tilted by half_angle from
    # vertical: dot == cos(2 * half_angle), so the verdict flips past 60 degrees
    # for threshold 0.5 (cos(120 deg) == -0.5).
    [(0.0, False), (45.0, False), (59.0, False), (61.0, True), (90.0, True)],
)
def test_threshold_is_an_angle_between_the_two_sides(half_angle_deg, expected):
    a = np.radians(half_angle_deg)
    left = np.array([np.sin(a), 0.0, -np.cos(a)])
    right = np.array([-np.sin(a), 0.0, -np.cos(a)])
    assert bool(opposing_normals_ok(left, right, threshold=0.5)) is expected


def test_threshold_minus_one_accepts_any_bilateral_contact():
    """The escape hatch back to the pre-0.4.14 contact-only predicate."""
    assert bool(opposing_normals_ok(_DOWN, _DOWN, threshold=-1.0))


def test_zero_vector_never_grasps():
    """A side with no contact contributes a zero resultant and cannot oppose."""
    assert not bool(opposing_normals_ok(np.zeros(3), _DOWN, threshold=0.3))


def test_batched_numpy_matches_scalar_calls():
    grip = np.stack([_UP, _DOWN, np.zeros(3)])
    jaw = np.stack([_DOWN, _DOWN, _DOWN])
    got = opposing_normals_ok(grip, jaw, threshold=0.3)
    assert got.tolist() == [True, False, False]


def test_torch_and_numpy_agree():
    """Tensor and array paths are the same function, per the tensor-friendly rule."""
    torch = pytest.importorskip("torch")
    grip = np.stack([_UP, _DOWN])
    jaw = np.stack([_DOWN, _DOWN])
    np_out = opposing_normals_ok(grip, jaw, threshold=0.3)
    torch_out = opposing_normals_ok(
        torch.tensor(grip, dtype=torch.float64),
        torch.tensor(jaw, dtype=torch.float64),
        threshold=0.3,
    )
    assert torch_out.tolist() == np_out.tolist()
