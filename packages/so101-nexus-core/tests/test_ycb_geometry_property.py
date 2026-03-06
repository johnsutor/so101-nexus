from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

import so101_nexus_core.ycb_geometry as ycb_geometry


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
