"""Tests for camera_utils module."""

import numpy as np
import pytest

from so101_nexus_core.camera_utils import (
    _scene_bounds,
    compute_angled_camera_params,
    compute_overhead_camera_params,
)


class TestSceneBounds:
    def test_default_bounds(self):
        x_min, x_max, y_min, y_max = _scene_bounds((0.15, 0.0), 0.40, 0.10)
        assert x_min == pytest.approx(-0.10)
        assert x_max == pytest.approx(0.65)
        assert y_min == pytest.approx(-0.50)
        assert y_max == pytest.approx(0.50)

    def test_bounds_grow_with_radius(self):
        _, x1, _, y1 = _scene_bounds((0.15, 0.0), 0.30, 0.10)
        _, x2, _, y2 = _scene_bounds((0.15, 0.0), 0.60, 0.10)
        assert x2 > x1
        assert y2 > y1


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
        # Lookat should be at center of bounding box, shifted forward from origin.
        assert params["lookat"][0] > 0.0
        assert params["lookat"][2] == pytest.approx(0.0, abs=0.01)
        assert params["elevation"] == -90
        assert params["azimuth"] == 0

    def test_lookat_centered_on_scene(self):
        params = compute_overhead_camera_params(
            spawn_center=(0.15, 0.0),
            spawn_max_radius=0.40,
            margin=0.10,
        )
        # Scene X: [-0.10, 0.65] -> center = 0.275
        # Scene Y: [-0.50, 0.50] -> center = 0.0
        np.testing.assert_allclose(params["lookat"][:2], [0.275, 0.0], atol=1e-6)

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
        p1 = compute_overhead_camera_params(spawn_center=(0.15, 0.0), spawn_max_radius=0.40)
        p2 = compute_overhead_camera_params(spawn_center=(0.25, 0.1), spawn_max_radius=0.40)
        assert p2["lookat"][0] > p1["lookat"][0]
        assert p2["lookat"][1] > p1["lookat"][1]


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
