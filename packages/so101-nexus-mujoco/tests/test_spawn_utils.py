"""Unit tests for so101_nexus_mujoco.spawn_utils.sample_separated_positions."""

from __future__ import annotations

import math

import numpy as np
import pytest

from so101_nexus_mujoco.spawn_utils import random_yaw_quat, sample_separated_positions


def _make_rng(seed: int = 42) -> np.random.Generator:
    """Create a reproducible NumPy random generator for spawn sampling tests."""
    return np.random.default_rng(seed)


def test_return_type_and_shape():
    """Sampling should return one XY tuple per requested object."""
    rng = _make_rng()
    count = 4
    result = sample_separated_positions(
        rng=rng,
        count=count,
        min_r=0.1,
        max_r=0.3,
        angle_half=math.pi / 2,
        min_clearance=0.01,
        bounding_radii=[0.03] * count,
    )
    assert isinstance(result, list)
    assert len(result) == count
    for pos in result:
        assert isinstance(pos, tuple)
        assert len(pos) == 2
        assert isinstance(pos[0], float)
        assert isinstance(pos[1], float)


def test_separation_guarantee():
    """Widely-spaced arc; objects must respect the minimum separation contract."""
    rng = _make_rng(0)
    count = 3
    bounding_radii = [0.04, 0.04, 0.04]
    min_clearance = 0.02
    result = sample_separated_positions(
        rng=rng,
        count=count,
        min_r=0.05,
        max_r=0.50,
        angle_half=math.pi,
        min_clearance=min_clearance,
        bounding_radii=bounding_radii,
        max_attempts=500,
    )
    assert len(result) == count
    for i, (xi, yi) in enumerate(result):
        for j, (xj, yj) in enumerate(result):
            if i >= j:
                continue
            dist = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
            required = bounding_radii[i] + bounding_radii[j] + min_clearance
            assert dist >= required, (
                f"Positions {i} and {j} overlap: dist={dist:.4f} < required={required:.4f}"
            )


def test_positions_within_radius_bounds():
    """Every sampled point should remain inside the requested radial band."""
    rng = _make_rng(1)
    count = 5
    min_r, max_r = 0.10, 0.40
    result = sample_separated_positions(
        rng=rng,
        count=count,
        min_r=min_r,
        max_r=max_r,
        angle_half=math.pi / 2,
        min_clearance=0.005,
        bounding_radii=[0.02] * count,
        max_attempts=200,
    )
    for x, y in result:
        r = math.sqrt(x**2 + y**2)
        assert min_r <= r <= max_r, f"Radial distance {r:.4f} out of [{min_r}, {max_r}]"


def test_angle_bounds():
    """Every sampled point should remain inside the requested angular band."""
    rng = _make_rng(2)
    count = 6
    angle_half = math.pi / 4
    result = sample_separated_positions(
        rng=rng,
        count=count,
        min_r=0.10,
        max_r=0.50,
        angle_half=angle_half,
        min_clearance=0.005,
        bounding_radii=[0.02] * count,
        max_attempts=300,
    )
    for x, y in result:
        theta = math.atan2(y, x)
        assert -angle_half <= theta <= angle_half, (
            f"Angle {theta:.4f} rad outside [-{angle_half:.4f}, {angle_half:.4f}]"
        )


def test_count_zero_returns_empty():
    """Requesting zero objects should produce an empty result."""
    rng = _make_rng()
    result = sample_separated_positions(
        rng=rng,
        count=0,
        min_r=0.1,
        max_r=0.3,
        angle_half=math.pi / 2,
        min_clearance=0.01,
        bounding_radii=[],
    )
    assert result == []


def test_count_one_returns_single_position():
    """Requesting one object should still honor the radial bounds."""
    rng = _make_rng()
    result = sample_separated_positions(
        rng=rng,
        count=1,
        min_r=0.10,
        max_r=0.30,
        angle_half=math.pi / 2,
        min_clearance=0.01,
        bounding_radii=[0.05],
    )
    assert len(result) == 1
    x, y = result[0]
    r = math.sqrt(x**2 + y**2)
    assert 0.10 <= r <= 0.30


def test_fallback_still_returns_count_positions():
    """When max_attempts=1 and objects are too large to separate, the function
    must still return exactly `count` positions rather than raising.
    """
    rng = _make_rng(99)
    count = 5
    result = sample_separated_positions(
        rng=rng,
        count=count,
        min_r=0.05,
        max_r=0.10,
        angle_half=math.pi / 6,
        min_clearance=5.0,
        bounding_radii=[1.0] * count,
        max_attempts=1,
    )
    assert len(result) == count


def test_max_attempts_zero_raises():
    """max_attempts must be at least 1; passing 0 should raise ValueError."""
    rng = _make_rng(0)
    with pytest.raises(ValueError, match="max_attempts"):
        sample_separated_positions(
            rng=rng,
            count=2,
            min_r=0.1,
            max_r=0.3,
            angle_half=math.pi / 2,
            min_clearance=0.01,
            bounding_radii=[0.02, 0.02],
            max_attempts=0,
        )


def test_determinism_with_fixed_seed():
    """A fixed RNG seed should produce the same spawn sequence every time."""
    kwargs = dict(
        count=4,
        min_r=0.10,
        max_r=0.40,
        angle_half=math.pi / 2,
        min_clearance=0.02,
        bounding_radii=[0.04, 0.04, 0.04, 0.04],
        max_attempts=100,
    )
    result_a = sample_separated_positions(rng=_make_rng(7), **kwargs)
    result_b = sample_separated_positions(rng=_make_rng(7), **kwargs)
    assert result_a == result_b


class TestRandomYawQuat:
    """Tests for random_yaw_quat."""

    def _make_rng(self, seed: int = 0) -> np.random.Generator:
        return np.random.default_rng(seed)

    def test_returns_unit_quaternion(self):
        """Result should be a unit quaternion."""
        rng = self._make_rng()
        q = random_yaw_quat(rng)
        assert abs(np.linalg.norm(q) - 1.0) < 1e-9

    def test_shape_is_4(self):
        """Result should be shape (4,)."""
        rng = self._make_rng()
        q = random_yaw_quat(rng)
        assert q.shape == (4,)

    def test_zero_roll_and_pitch(self):
        """x and y components of the quaternion must be zero (pure yaw)."""
        rng = self._make_rng(42)
        for _ in range(50):
            q = random_yaw_quat(rng)
            assert abs(q[1]) < 1e-9, "x component should be zero"
            assert abs(q[2]) < 1e-9, "y component should be zero"

    def test_reproducible_with_seed(self):
        """Same seed produces same quaternion."""
        q1 = random_yaw_quat(self._make_rng(7))
        q2 = random_yaw_quat(self._make_rng(7))
        np.testing.assert_array_equal(q1, q2)
