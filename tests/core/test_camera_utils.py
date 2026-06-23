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
    @given(
        cx=finite_center,
        cy=finite_center,
        radius=positive_extent,
        margin=nonnegative_margin,
        elev=st.floats(min_value=-89.0, max_value=0.0, allow_nan=False, allow_infinity=False),
        az=st.floats(min_value=0.0, max_value=360.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200)
    def test_angled_camera_reuses_overhead_lookat_and_pulls_back(
        self, cx, cy, radius, margin, elev, az
    ):
        overhead = compute_overhead_camera_params(
            spawn_center=(cx, cy),
            spawn_max_radius=radius,
            margin=margin,
        )
        angled = compute_angled_camera_params(
            spawn_center=(cx, cy),
            spawn_max_radius=radius,
            margin=margin,
            elevation=elev,
            azimuth=az,
        )

        assert set(angled) == {"lookat", "distance", "elevation", "azimuth"}
        np.testing.assert_array_equal(angled["lookat"], overhead["lookat"])
        assert angled["distance"] == pytest.approx(overhead["distance"] * 1.2)
        assert angled["elevation"] == elev
        assert angled["azimuth"] == az
