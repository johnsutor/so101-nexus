"""Tests for camera_utils module."""

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from so101_nexus.camera_utils import (
    _scene_bounds,
    compute_angled_camera_params,
    compute_overhead_camera_params,
    compute_overhead_eye_target,
)

finite_center = st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False)
positive_extent = st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False)
nonnegative_margin = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
fov_deg = st.floats(min_value=1.0, max_value=170.0, allow_nan=False, allow_infinity=False)
aspect_ratio = st.floats(min_value=0.25, max_value=4.0, allow_nan=False, allow_infinity=False)


class TestSceneBounds:
    @given(cx=finite_center, cy=finite_center, radius=positive_extent, margin=nonnegative_margin)
    @settings(max_examples=200)
    def test_bounds_match_spawn_arc_formula(self, cx, cy, radius, margin):
        x_min, x_max, y_min, y_max = _scene_bounds((cx, cy), radius, margin)

        assert x_min == pytest.approx(-margin)
        assert x_max == pytest.approx(cx + radius + margin)
        assert y_min == pytest.approx(cy - radius - margin)
        assert y_max == pytest.approx(cy + radius + margin)


class TestComputeOverheadCameraParams:
    @given(cx=finite_center, cy=finite_center, radius=positive_extent, margin=nonnegative_margin)
    @settings(max_examples=200)
    def test_default_framing_and_eye_target_agree(self, cx, cy, radius, margin):
        kwargs = {"spawn_center": (cx, cy), "spawn_max_radius": radius, "margin": margin}
        params = compute_overhead_camera_params(**kwargs)
        eye, target = compute_overhead_eye_target(**kwargs)
        assert math.isfinite(params["distance"])
        assert params["distance"] > 0.0
        assert params["elevation"] == -90
        assert params["azimuth"] == 0
        assert eye[0] == target[0]
        assert eye[1] == target[1]
        assert eye[2] > target[2]
        assert target[2] == 0.0
        np.testing.assert_array_equal(target, params["lookat"])
        assert eye[2] == params["distance"]

    @given(
        cx=finite_center,
        cy=finite_center,
        radius=positive_extent,
        margin=nonnegative_margin,
        fov=fov_deg,
        aspect=aspect_ratio,
    )
    @settings(max_examples=200)
    def test_overhead_camera_centers_and_bounds_scene(self, cx, cy, radius, margin, fov, aspect):
        params = compute_overhead_camera_params(
            spawn_center=(cx, cy),
            spawn_max_radius=radius,
            margin=margin,
            fov_deg=fov,
            aspect=aspect,
        )

        assert set(params) == {"lookat", "distance", "elevation", "azimuth"}
        np.testing.assert_allclose(params["lookat"], [(cx + radius) / 2.0, cy, 0.0], atol=1e-12)
        assert math.isfinite(params["distance"])
        assert params["distance"] > 0.0
        assert params["elevation"] == -90
        assert params["azimuth"] == 0

    @given(
        cx=finite_center,
        cy=finite_center,
        radius=positive_extent,
        extra_radius=positive_extent,
        margin=nonnegative_margin,
    )
    @settings(max_examples=100)
    def test_wider_spawn_gives_larger_distance(self, cx, cy, radius, extra_radius, margin):
        narrow = compute_overhead_camera_params(
            spawn_center=(cx, cy),
            spawn_max_radius=radius,
            margin=margin,
        )
        wide = compute_overhead_camera_params(
            spawn_center=(cx, cy),
            spawn_max_radius=radius + extra_radius,
            margin=margin,
        )

        assert wide["distance"] > narrow["distance"]

    @given(
        cx=finite_center,
        cy=finite_center,
        dx=finite_center,
        dy=finite_center,
        radius=positive_extent,
        margin=nonnegative_margin,
    )
    @settings(max_examples=100)
    def test_offset_center_shifts_lookat(self, cx, cy, dx, dy, radius, margin):
        base = compute_overhead_camera_params(
            spawn_center=(cx, cy),
            spawn_max_radius=radius,
            margin=margin,
        )
        shifted = compute_overhead_camera_params(
            spawn_center=(cx + dx, cy + dy),
            spawn_max_radius=radius,
            margin=margin,
        )

        np.testing.assert_allclose(
            shifted["lookat"] - base["lookat"],
            [dx / 2.0, dy, 0.0],
            atol=1e-12,
        )


class TestComputeAngledCameraParams:
    @pytest.mark.parametrize("azimuth", [90.0, 120.0, 160.0, 200.0])
    @pytest.mark.parametrize("elevation", [-90.0, -45.0, -20.0])
    @pytest.mark.parametrize("aspect", [0.75, 4 / 3, 2.0])
    def test_workspace_and_robot_fit_perspective_frame(self, azimuth, elevation, aspect):
        params = compute_angled_camera_params(
            spawn_max_radius=0.3, azimuth=azimuth, elevation=elevation, aspect=aspect
        )
        angles = np.linspace(-np.pi / 2, np.pi / 2, 361)
        workspace = np.column_stack(
            (0.15 + 0.3 * np.cos(angles), 0.3 * np.sin(angles), np.zeros_like(angles))
        )
        points = np.concatenate((workspace, [[0, 0, 0], [0, 0, 0.4], [0.25, 0, 0.25]]))
        azimuth_rad, elevation_rad = np.radians([azimuth, elevation])
        ca, sa = np.cos(azimuth_rad), np.sin(azimuth_rad)
        ce, se = np.cos(elevation_rad), np.sin(elevation_rad)
        forward = np.array([ca * ce, sa * ce, se])
        right = np.array([sa, -ca, 0])
        up = np.array([-ca * se, -sa * se, ce])
        offsets = points - params["lookat"]
        depth = params["distance"] + offsets @ forward
        assert (depth > 0).all()
        tan_vfov = np.tan(np.radians(22.5))
        assert (np.abs(offsets @ up) <= depth * tan_vfov + 1e-12).all()
        assert (np.abs(offsets @ right) <= depth * tan_vfov * aspect + 1e-12).all()

    def test_default_frame_is_closer_and_targets_above_table(self):
        params = compute_angled_camera_params(spawn_max_radius=0.3)
        overhead = compute_overhead_camera_params(spawn_max_radius=0.3)
        assert params["distance"] < overhead["distance"] * 1.2
        assert params["lookat"][2] > 0

    def test_narrower_spawn_arc_has_tighter_frame(self):
        narrow = compute_angled_camera_params(spawn_angle_half_range_deg=30)
        wide = compute_angled_camera_params(spawn_angle_half_range_deg=180)
        assert narrow["distance"] < wide["distance"]

    @pytest.mark.parametrize("scalar_angle", [None, "azimuth", "elevation"])
    def test_batched_tensor_angles_match_scalar_parameters(self, scalar_angle):
        torch = pytest.importorskip("torch")
        azimuth = torch.tensor([90.0, 160.0, 200.0], dtype=torch.float64)
        elevation = torch.tensor([-20.0, -30.0, -45.0], dtype=torch.float64)
        if scalar_angle == "azimuth":
            azimuth[:] = 160
        elif scalar_angle == "elevation":
            elevation[:] = -30
        params = compute_angled_camera_params(
            azimuth=160.0 if scalar_angle == "azimuth" else azimuth,
            elevation=-30.0 if scalar_angle == "elevation" else elevation,
        )
        expected = [
            compute_angled_camera_params(azimuth=float(az), elevation=float(el))["distance"]
            for az, el in zip(azimuth, elevation, strict=True)
        ]
        torch.testing.assert_close(params["distance"], torch.tensor(expected, dtype=torch.float64))

    @given(
        cx=finite_center,
        cy=finite_center,
        radius=positive_extent,
        margin=nonnegative_margin,
        fov=fov_deg,
        aspect=aspect_ratio,
        half_range=st.floats(min_value=1, max_value=180),
        azimuth=st.floats(min_value=0, max_value=360),
        elevation=st.floats(min_value=-90, max_value=-1),
    )
    @settings(max_examples=200)
    def test_arbitrary_workspace_fits_frame(
        self, cx, cy, radius, margin, fov, aspect, half_range, azimuth, elevation
    ):
        params = compute_angled_camera_params(
            spawn_center=(cx, cy),
            spawn_max_radius=radius,
            margin=margin,
            fov_deg=fov,
            aspect=aspect,
            azimuth=azimuth,
            elevation=elevation,
            spawn_angle_half_range_deg=half_range,
        )
        angles = np.radians(np.linspace(-half_range, half_range, 181))
        points = np.column_stack(
            (cx + radius * np.cos(angles), cy + radius * np.sin(angles), np.zeros_like(angles))
        )
        points = np.concatenate((points, [[0, 0, 0], [0, 0, 0.4], [0.25, 0, 0.25]]))
        az, el = np.radians([azimuth, elevation])
        forward = np.array([np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)])
        right = np.array([np.sin(az), -np.cos(az), 0])
        up = np.array([-np.cos(az) * np.sin(el), -np.sin(az) * np.sin(el), np.cos(el)])
        offsets = points - params["lookat"]
        depth = params["distance"] + offsets @ forward
        tangent = np.tan(np.radians(fov / 2))
        assert np.isfinite(params["distance"])
        assert (depth > 0).all()
        assert (abs(offsets @ right) <= depth * tangent * aspect + 1e-10).all()
        assert (abs(offsets @ up) <= depth * tangent + 1e-10).all()


@pytest.mark.parametrize("azimuth,elevation", [(160, -30), (120, -45), (200, -20)])
def test_square_spawn_corners_fit_side_camera(azimuth, elevation):
    params = compute_angled_camera_params(
        spawn_max_radius=0.4, spawn_half_size=0.4, azimuth=azimuth, elevation=elevation
    )
    points = np.array([[x, y, 0.02] for x in (-0.25, 0.55) for y in (-0.4, 0.4)])
    az, el = np.radians([azimuth, elevation])
    forward = np.array([np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)])
    right = np.array([np.sin(az), -np.cos(az), 0])
    up = np.array([-np.cos(az) * np.sin(el), -np.sin(az) * np.sin(el), np.cos(el)])
    offsets = points - params["lookat"]
    depth = params["distance"] + offsets @ forward
    tangent = np.tan(np.radians(22.5))
    assert (abs(offsets @ right) <= depth * tangent * 4 / 3).all()
    assert (abs(offsets @ up) <= depth * tangent).all()
