"""Tests for the backend-shared gaze predicate primitives."""

import math

import numpy as np
import pytest

from so101_nexus.gaze import direction_to_object, gaze_angle_rad, gaze_cosine, object_in_view

_FORWARD = np.array([1.0, 0.0, 0.0])


def _angle_between(axis, camera_pos, object_pos):
    return float(gaze_angle_rad(gaze_cosine(axis, direction_to_object(camera_pos, object_pos))))


def test_object_on_the_optical_axis_has_zero_angle():
    angle = _angle_between(_FORWARD, np.zeros(3), np.array([0.5, 0.0, 0.0]))
    assert angle == pytest.approx(0.0, abs=1e-6)


def test_object_behind_the_camera_is_half_a_turn_away():
    angle = _angle_between(_FORWARD, np.zeros(3), np.array([-0.5, 0.0, 0.0]))
    assert angle == pytest.approx(math.pi, abs=1e-6)


@pytest.mark.parametrize("offset_deg", [5.0, 30.0, 91.0, 179.0])
def test_angle_recovers_the_geometry_it_was_built_from(offset_deg):
    rad = math.radians(offset_deg)
    obj = np.array([math.cos(rad), math.sin(rad), 0.0]) * 0.3
    assert _angle_between(_FORWARD, np.zeros(3), obj) == pytest.approx(rad, abs=1e-6)


def test_distance_does_not_change_the_angle():
    near = _angle_between(_FORWARD, np.zeros(3), np.array([0.1, 0.05, 0.0]))
    far = _angle_between(_FORWARD, np.zeros(3), np.array([1.0, 0.5, 0.0]))
    assert near == pytest.approx(far, abs=1e-6)


def test_in_view_flips_at_the_half_fov_boundary():
    half_fov = math.radians(24.25)  # the SO-101 wrist camera's 48.5 degree fovy
    assert bool(object_in_view(half_fov - 1e-6, half_fov))
    assert not bool(object_in_view(half_fov + 1e-6, half_fov))


def test_coincident_camera_and_object_do_not_divide_by_zero():
    direction = direction_to_object(np.zeros(3), np.zeros(3))
    assert np.all(np.isfinite(direction))
    assert float(np.linalg.norm(direction)) == pytest.approx(0.0)


def test_cosine_is_clamped_for_arccos():
    # Slightly non-unit inputs (float round-off on a normalized vector) must not
    # push arccos out of its domain.
    axis = _FORWARD * (1.0 + 1e-7)
    assert np.isfinite(gaze_angle_rad(gaze_cosine(axis, _FORWARD * (1.0 + 1e-7))))


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_torch_and_numpy_agree(dtype_name):
    """Tensor and array paths are the same function, per the tensor-friendly rule."""
    torch = pytest.importorskip("torch")
    dtype = getattr(torch, dtype_name)
    cam = np.stack([np.zeros(3), np.array([0.0, 0.1, 0.0])])
    obj = np.stack([np.array([0.3, 0.1, 0.0]), np.array([-0.2, 0.0, 0.1])])
    axis = np.stack([_FORWARD, _FORWARD])
    half_fov = math.radians(30.0)

    np_angle = gaze_angle_rad(gaze_cosine(axis, direction_to_object(cam, obj)))
    torch_angle = gaze_angle_rad(
        gaze_cosine(
            torch.tensor(axis, dtype=dtype),
            direction_to_object(torch.tensor(cam, dtype=dtype), torch.tensor(obj, dtype=dtype)),
        )
    )
    np.testing.assert_allclose(torch_angle.numpy(), np_angle, rtol=0, atol=1e-6)
    assert (
        object_in_view(torch_angle, half_fov).tolist()
        == object_in_view(np_angle, half_fov).tolist()
    )


def test_per_world_half_fov_is_applied_elementwise():
    """Wrist-camera randomization gives each world its own boundary."""
    torch = pytest.importorskip("torch")
    angle = torch.tensor([0.3, 0.3])
    half_fov = torch.tensor([0.2, 0.4])
    assert object_in_view(angle, half_fov).tolist() == [False, True]
