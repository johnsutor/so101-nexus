"""Tests for camera_utils module."""

import numpy as np
import pytest

from so101_nexus_core.camera_utils import (
    compute_angled_camera_params,
    compute_overhead_camera_params,
)


class TestComputeOverheadCameraParams:
    def test_default_spawn_config(self):
        params = compute_overhead_camera_params(
            spawn_center=(0.15, 0.0),
            spawn_max_radius=0.40,
        )
        assert "lookat" in params
        assert "distance" in params
        assert "elevation" in params
        assert "azimuth" in params
        np.testing.assert_allclose(params["lookat"][:2], [0.15, 0.0], atol=1e-6)
        assert params["lookat"][2] == pytest.approx(0.0, abs=0.01)
        assert params["elevation"] == -90
        assert params["azimuth"] == 0

    def test_distance_covers_spawn_area(self):
        params = compute_overhead_camera_params(
            spawn_center=(0.15, 0.0),
            spawn_max_radius=0.40,
        )
        assert params["distance"] > 0.40

    def test_wider_spawn_gives_larger_distance(self):
        narrow = compute_overhead_camera_params(
            spawn_center=(0.15, 0.0),
            spawn_max_radius=0.30,
        )
        wide = compute_overhead_camera_params(
            spawn_center=(0.15, 0.0),
            spawn_max_radius=0.60,
        )
        assert wide["distance"] > narrow["distance"]

    def test_offset_center_shifts_lookat(self):
        params = compute_overhead_camera_params(
            spawn_center=(0.25, 0.1),
            spawn_max_radius=0.40,
        )
        np.testing.assert_allclose(params["lookat"][:2], [0.25, 0.1], atol=1e-6)

    def test_origin_center(self):
        params = compute_overhead_camera_params(
            spawn_center=(0.0, 0.0),
            spawn_max_radius=0.40,
        )
        np.testing.assert_allclose(params["lookat"][:2], [0.0, 0.0], atol=1e-6)
        assert params["distance"] > 0.40


class TestComputeAngledCameraParams:
    def test_returns_expected_keys(self):
        params = compute_angled_camera_params(
            spawn_center=(0.15, 0.0),
            spawn_max_radius=0.40,
        )
        assert "lookat" in params
        assert "distance" in params
        assert "elevation" in params
        assert "azimuth" in params

    def test_elevation_is_angled_not_overhead(self):
        params = compute_angled_camera_params(
            spawn_center=(0.15, 0.0),
            spawn_max_radius=0.40,
        )
        assert -60 < params["elevation"] < -10

    def test_distance_larger_than_overhead(self):
        overhead = compute_overhead_camera_params(
            spawn_center=(0.15, 0.0),
            spawn_max_radius=0.40,
        )
        angled = compute_angled_camera_params(
            spawn_center=(0.15, 0.0),
            spawn_max_radius=0.40,
        )
        assert angled["distance"] > overhead["distance"]
