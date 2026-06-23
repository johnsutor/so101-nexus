"""Unit tests for so101_nexus.mujoco.spawn_utils.sample_separated_positions."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from so101_nexus.mujoco import spawn_utils
from so101_nexus.mujoco.spawn_utils import random_yaw_quat, sample_separated_positions


def _make_rng(seed: int = 42) -> np.random.Generator:
    """Create a reproducible NumPy random generator for spawn sampling tests."""
    return np.random.default_rng(seed)


@given(
    count=st.integers(min_value=0, max_value=5),
    min_r=st.floats(min_value=0.05, max_value=0.2, allow_nan=False, allow_infinity=False),
    max_r_delta=st.floats(min_value=0.05, max_value=0.5, allow_nan=False, allow_infinity=False),
    angle_half=st.floats(min_value=0.1, max_value=math.pi, allow_nan=False, allow_infinity=False),
    center_x=st.floats(min_value=-0.5, max_value=0.5, allow_nan=False, allow_infinity=False),
    center_y=st.floats(min_value=-0.5, max_value=0.5, allow_nan=False, allow_infinity=False),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=100)
def test_sample_separated_positions_contract(
    count, min_r, max_r_delta, angle_half, center_x, center_y, seed
):
    """Sampling returns finite XY positions inside the requested polar arc."""
    rng = _make_rng(seed)
    max_r = min_r + max_r_delta
    center = (center_x, center_y)
    result = sample_separated_positions(
        rng=rng,
        count=count,
        min_r=min_r,
        max_r=max_r,
        angle_half=angle_half,
        min_clearance=0.01,
        bounding_radii=[0.02] * count,
        center=center,
    )

    assert isinstance(result, list)
    assert len(result) == count
    for x, y in result:
        assert isinstance(x, float)
        assert isinstance(y, float)
        assert np.isfinite(x)
        assert np.isfinite(y)
        dx = x - center_x
        dy = y - center_y
        radius = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        assert min_r <= radius <= max_r
        assert -angle_half <= theta <= angle_half


@given(
    count=st.integers(min_value=2, max_value=4),
    radius=st.floats(min_value=0.005, max_value=0.02, allow_nan=False, allow_infinity=False),
    clearance=st.floats(min_value=0.0, max_value=0.01, allow_nan=False, allow_infinity=False),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=100)
def test_sample_separated_positions_respects_clearance_when_feasible(
    count, radius, clearance, seed
):
    """In an intentionally wide arc, objects respect pairwise clearance."""
    radii = [radius] * count
    result = sample_separated_positions(
        rng=_make_rng(seed),
        count=count,
        min_r=0.10,
        max_r=1.00,
        angle_half=math.pi,
        min_clearance=clearance,
        bounding_radii=radii,
        max_attempts=1000,
    )

    for i, (xi, yi) in enumerate(result):
        for j, (xj, yj) in enumerate(result):
            if i >= j:
                continue
            dist = math.hypot(xi - xj, yi - yj)
            assert dist >= radii[i] + radii[j] + clearance


@given(
    count=st.integers(min_value=0, max_value=5),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=100)
def test_sample_separated_positions_deterministic_for_seed(count, seed):
    """A fixed RNG seed produces the same spawn sequence."""
    kwargs = {
        "count": count,
        "min_r": 0.10,
        "max_r": 0.40,
        "angle_half": math.pi / 2,
        "min_clearance": 0.02,
        "bounding_radii": [0.01] * count,
        "max_attempts": 100,
    }
    result_a = sample_separated_positions(rng=_make_rng(seed), **kwargs)
    result_b = sample_separated_positions(rng=_make_rng(seed), **kwargs)
    assert result_a == result_b


def test_fallback_still_returns_count_positions():
    """Infeasible separation falls back instead of raising."""
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
    """max_attempts must be at least 1."""
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


class _FakeMeshModel:
    def __init__(self, geom_type: int):
        self.geom_type = np.array([geom_type], dtype=np.int32)
        self.geom_dataid = np.array([0], dtype=np.int32)
        self.mesh_vertadr = np.array([0], dtype=np.int32)
        self.mesh_vertnum = np.array([3], dtype=np.int32)
        self.mesh_vert = np.array(
            [
                [-0.1, 0.0, -0.02],
                [0.1, 0.0, 0.03],
                [0.0, 0.2, 0.01],
            ],
            dtype=np.float64,
        )


class _FakeGeomData:
    def __init__(self):
        self.qpos = np.zeros(7, dtype=np.float64)
        self.geom_xpos = np.array([[0.0, 0.0, 0.5]], dtype=np.float64)
        self.geom_xmat = np.eye(3, dtype=np.float64).reshape(1, 9)


def test_mesh_geom_world_min_z_uses_compiled_mesh_transform():
    import mujoco

    model = _FakeMeshModel(mujoco.mjtGeom.mjGEOM_MESH)
    data = _FakeGeomData()

    min_z = spawn_utils.mesh_geom_world_min_z(model, data, 0)

    assert min_z == pytest.approx(0.48)


def test_mesh_geom_world_min_z_rejects_non_mesh_geom():
    import mujoco

    model = _FakeMeshModel(mujoco.mjtGeom.mjGEOM_BOX)
    data = _FakeGeomData()

    with pytest.raises(ValueError, match="mesh geom"):
        spawn_utils.mesh_geom_world_min_z(model, data, 0)


def test_align_freejoint_geom_to_floor_sets_spawn_height(monkeypatch):
    import mujoco

    model = _FakeMeshModel(mujoco.mjtGeom.mjGEOM_MESH)
    data = _FakeGeomData()

    def _fake_forward(_model, fake_data):
        fake_data.geom_xpos[0] = fake_data.qpos[:3]

    monkeypatch.setattr(mujoco, "mj_forward", _fake_forward)

    spawn_z = spawn_utils.align_freejoint_geom_to_floor(
        model,
        data,
        qpos_addr=0,
        geom_id=0,
        xy=(0.2, -0.1),
        quat=np.array([1.0, 0.0, 0.0, 0.0]),
        margin=0.005,
    )

    assert spawn_z == pytest.approx(0.025)
    np.testing.assert_allclose(data.qpos[:3], [0.2, -0.1, 0.025])
    np.testing.assert_allclose(data.qpos[3:7], [1.0, 0.0, 0.0, 0.0])


@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
@settings(max_examples=200)
def test_random_yaw_quat_contract(seed):
    rng = np.random.default_rng(seed)
    q = random_yaw_quat(rng)

    assert q.shape == (4,)
    np.testing.assert_allclose(np.linalg.norm(q), 1.0, atol=1e-6)
    assert q[1] == 0.0
    assert q[2] == 0.0
    np.testing.assert_array_equal(q, random_yaw_quat(np.random.default_rng(seed)))
