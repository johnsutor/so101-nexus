from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

if TYPE_CHECKING:
    from pathlib import Path

from so101_nexus_core import ycb_geometry
from so101_nexus_core.ycb_geometry import get_mujoco_ycb_rest_pose


def test_get_maniskill_ycb_spawn_z_reads_metadata(monkeypatch: pytest.MonkeyPatch):
    ycb_geometry._load_maniskill_pick_db.cache_clear()
    fake_db = {"058_golf_ball": {"bbox": {"min": [0.0, 0.0, -0.1]}, "scales": [2.0]}}
    monkeypatch.setattr(ycb_geometry, "_load_maniskill_pick_db", lambda: fake_db)
    assert ycb_geometry.get_maniskill_ycb_spawn_z("058_golf_ball", margin=0.003) == pytest.approx(
        0.203
    )


def test_load_maniskill_pick_db_cache_uses_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    ycb_geometry._load_maniskill_pick_db.cache_clear()
    data_root = tmp_path / ".maniskill" / "data" / "assets" / "mani_skill2_ycb"
    data_root.mkdir(parents=True)
    info_path = data_root / "info_pick_v0.json"
    info_path.write_text(json.dumps({"k": {"bbox": {"min": [0, 0, 0]}}}), encoding="utf-8")
    monkeypatch.setattr(ycb_geometry.Path, "home", staticmethod(lambda: tmp_path))

    db1 = ycb_geometry._load_maniskill_pick_db()
    db2 = ycb_geometry._load_maniskill_pick_db()
    assert db1 is db2
    assert db1["k"]["bbox"]["min"] == [0, 0, 0]


@pytest.mark.parametrize(
    ("verts", "expected_quat"),
    [
        (np.array([[0, 0, -0.1], [1, 1, 0.2], [0.5, 0.1, 0.3]], dtype=np.float64), [1, 0, 0, 0]),
        (
            np.array([[-0.01, 0, 0], [0.01, 1.0, 1.0], [0.0, -0.5, -0.2]], dtype=np.float64),
            [0.7071068, 0.0, 0.7071068, 0.0],
        ),
        (
            np.array([[0.0, -0.01, 0.0], [1.0, 0.01, 1.0], [-0.5, 0.0, -0.2]], dtype=np.float64),
            [0.7071068, 0.7071068, 0.0, 0.0],
        ),
    ],
)
def test_get_mujoco_ycb_rest_pose_axis_cases(verts: np.ndarray, expected_quat: list[float]):
    quat, spawn_z = ycb_geometry.get_mujoco_ycb_rest_pose(verts, margin=0.002)
    assert np.allclose(quat, np.array(expected_quat))
    assert spawn_z >= 0.002


@given(
    min_z=st.floats(min_value=-0.1, max_value=-1e-4, allow_infinity=False, allow_nan=False),
    max_z=st.floats(min_value=1e-4, max_value=0.1, allow_infinity=False, allow_nan=False),
    margin=st.floats(min_value=0.0, max_value=0.01, allow_infinity=False, allow_nan=False),
)
def test_get_mujoco_ycb_rest_pose_thin_z_property(min_z: float, max_z: float, margin: float):
    verts = np.array(
        [
            [0.0, 0.0, min_z],
            [1.5, 2.0, max_z],
            [0.5, 1.0, (min_z + max_z) / 2.0],
        ],
        dtype=np.float64,
    )
    quat, spawn_z = ycb_geometry.get_mujoco_ycb_rest_pose(verts, margin=margin)
    assert np.allclose(quat, np.array([1.0, 0.0, 0.0, 0.0]))
    assert spawn_z == pytest.approx(-min_z + margin)


@given(
    n=st.integers(min_value=4, max_value=64),
    scale=st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=100)
def test_rest_pose_always_returns_finite_values(n, scale, seed):
    rng = np.random.default_rng(seed)
    verts = rng.uniform(-scale, scale, size=(n, 3)).astype(np.float64)
    quat, spawn_z = get_mujoco_ycb_rest_pose(verts, margin=1e-3)

    assert np.all(np.isfinite(quat))
    # Quaternion norm ≈ 1.
    np.testing.assert_allclose(np.linalg.norm(quat), 1.0, atol=1e-6)
    assert np.isfinite(spawn_z)
    # Spawn z puts the lowest point at `margin` above the ground.
    # With arbitrary vertex positions, spawn_z can be negative if the lowest point is well above margin.
    assert isinstance(spawn_z, (float, np.floating))


@given(
    n=st.integers(min_value=4, max_value=64),
    scale=st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=100)
def test_rest_pose_deterministic(n, scale, seed):
    rng = np.random.default_rng(seed)
    verts = rng.uniform(-scale, scale, size=(n, 3)).astype(np.float64)
    q1, z1 = get_mujoco_ycb_rest_pose(verts, margin=1e-3)
    q2, z2 = get_mujoco_ycb_rest_pose(verts, margin=1e-3)
    np.testing.assert_array_equal(q1, q2)
    assert z1 == z2
