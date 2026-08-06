from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from so101_nexus import ycb_geometry
from so101_nexus.ycb_geometry import get_mujoco_ycb_rest_pose


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
    # Spawn z corresponds to -min-of-thinnest-axis + margin; may be negative
    # when the input happens to put all vertices above the origin on that axis.
    assert isinstance(spawn_z, float)


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


@pytest.mark.parametrize("thin_axis", [0, 1, 2])
@pytest.mark.parametrize("offset", [0.0, 0.11])
def test_spawn_z_matches_the_returned_rotation(thin_axis: int, offset: float):
    """``spawn_z`` must be the clearance of the mesh under ``quat``, not its mirror.

    Regression: the thin-X branch measured the height of the inverse rotation.
    That is invisible for vertices centered on the origin (the two agree by
    symmetry) and buries the object below the floor once they are offset, which
    is what the body-frame vertices from ``extract_object_slots`` are.
    """
    import mujoco

    half = [0.03, 0.03, 0.03]
    half[thin_axis] = 0.005
    corners = np.array(
        [
            [sx * half[0], sy * half[1], sz * half[2]]
            for sx in (-1, 1)
            for sy in (-1, 1)
            for sz in (-1, 1)
        ]
    )
    verts = corners + np.array([offset, offset, offset])

    quat, spawn_z = get_mujoco_ycb_rest_pose(verts, margin=0.002)

    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, quat)
    rotated_z = (verts @ rot.reshape(3, 3).T)[:, 2]
    assert spawn_z == pytest.approx(-rotated_z.min() + 0.002, abs=1e-6)
