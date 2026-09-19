"""Tests for the configurable render camera (overhead vs side view).

The side camera is a visualization-only view: it drives ``render_mode``
output and never appears in the observation space. Render-calling tests
skip when no GL context is available (headless CI without EGL).
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("so101_nexus.mujoco")

import mujoco

from so101_nexus import RenderConfig, TouchConfig
from so101_nexus.camera_utils import compute_angled_camera_params
from so101_nexus.observations import JointPositions, OverheadCamera

ENV_ID = "MuJoCoTouch-v1"


def _render_or_skip(env) -> np.ndarray:
    """Render one frame, skipping the test when no GL context is available."""
    try:
        frame = env.render()
    except (mujoco.FatalError, RuntimeError) as exc:
        msg = str(exc).lower()
        if any(k in msg for k in ("egl", "opengl", "gl ", "render", "context", "window")):
            pytest.skip(f"offscreen render unavailable in this environment: {exc}")
        raise
    assert frame is not None
    return np.asarray(frame)


def test_side_render_camera_params_follow_config():
    """The free render camera adopts the configured side view (no dead knobs)."""
    config = TouchConfig(
        render=RenderConfig(width=96, height=64, camera="side", side_azimuth_deg=120.0)
    )
    env = gym.make(ENV_ID, config=config, render_mode="rgb_array")
    try:
        env.reset(seed=0)
        frame = _render_or_skip(env)
        assert frame.shape == (64, 96, 3)
        assert frame.dtype == np.uint8

        cam = env.unwrapped._render_cam
        expected = compute_angled_camera_params(
            spawn_center=config.spawn_center,
            spawn_max_radius=config.spawn_max_radius,
            elevation=config.render.side_elevation_deg,
            azimuth=config.render.side_azimuth_deg,
            aspect=config.render.width / config.render.height,
        )
        assert cam.azimuth == pytest.approx(120.0)
        assert cam.elevation == pytest.approx(expected["elevation"])
        assert cam.distance == pytest.approx(expected["distance"])
        np.testing.assert_allclose(cam.lookat, expected["lookat"])
    finally:
        env.close()


def test_side_and_overhead_frames_differ():
    """Selecting the side camera actually switches the rendered viewpoint."""
    frames = {}
    for camera in ("overhead", "side"):
        config = TouchConfig(render=RenderConfig(width=96, height=64, camera=camera))
        env = gym.make(ENV_ID, config=config, render_mode="rgb_array")
        try:
            env.reset(seed=0)
            frames[camera] = _render_or_skip(env)
        finally:
            env.close()
    assert not np.array_equal(frames["overhead"], frames["side"])


@pytest.mark.parametrize(
    "observations",
    [None, [JointPositions(), OverheadCamera(width=64, height=48)]],
    ids=["state_only", "with_overhead_obs"],
)
def test_render_camera_does_not_affect_observation_space(observations):
    """The side camera is render-only: observation spaces are identical."""
    spaces = []
    for camera in ("overhead", "side"):
        config = TouchConfig(
            render=RenderConfig(camera=camera),
            observations=observations,
        )
        env = gym.make(ENV_ID, config=config)
        try:
            spaces.append(env.observation_space)
            obs, _ = env.reset(seed=0)
            if isinstance(obs, dict):
                assert not any("side" in key for key in obs)
        finally:
            env.close()
    assert spaces[0] == spaces[1]


@pytest.mark.parametrize("env_id", sorted(key for key in gym.registry if key.startswith("MuJoCo")))
@pytest.mark.parametrize("render_mode", ["rgb_array", "depth_array"])
def test_random_side_camera_all_tasks(env_factory, env_id, render_mode):
    task, version = env_id.removeprefix("MuJoCo").split("-v")
    env = env_factory(task=task, version=int(version), render_mode=render_mode)
    env.unwrapped.config.render = RenderConfig(
        width=64,
        height=48,
        camera="side",
        side_azimuth_range_deg=(90.0, 180.0),
        side_elevation_range_deg=(-60.0, -20.0),
        side_distance_range=(0.8, 1.2),
    )

    def pose():
        _render_or_skip(env)
        cam = env.unwrapped._render_cam
        return np.array([cam.azimuth, cam.elevation, cam.distance])

    env.reset(seed=42)
    first = pose()
    assert np.all(first >= [90.0, -60.0, 0.8])
    assert np.all(first <= [180.0, -20.0, 1.2])
    env.step(np.zeros(env.action_space.shape, dtype=env.action_space.dtype))
    np.testing.assert_array_equal(pose(), first)
    env.reset(seed=42)
    np.testing.assert_array_equal(pose(), first)
    env.reset()
    assert np.all(pose() != first)


@pytest.mark.parametrize("camera", ["side", "overhead"])
def test_fixed_side_ranges_and_overhead_rng(env_factory, camera):
    env = env_factory()
    base = env.unwrapped
    env.reset(seed=12)
    expected_rng = base.np_random.bit_generator.state
    base.config.render = RenderConfig(
        camera=camera,
        side_azimuth_range_deg=(123.0, 123.0),
        side_elevation_range_deg=(-45.0, -45.0),
        side_distance_range=(0.9, 0.9),
    )
    env.reset(seed=12)
    assert base.np_random.bit_generator.state == expected_rng
    params = base._render_camera_params()
    if camera == "side":
        assert [params[k] for k in ("azimuth", "elevation", "distance")] == [123.0, -45.0, 0.9]
    else:
        assert params["elevation"] == -90.0


@pytest.mark.parametrize(
    "field, key, value",
    [
        ("side_azimuth_range_deg", "azimuth", 100.0),
        ("side_elevation_range_deg", "elevation", -60.0),
        ("side_distance_range", "distance", 1.0),
    ],
)
def test_individual_side_range_updates_open_viewer(env_factory, field, key, value):
    from contextlib import nullcontext
    from unittest.mock import Mock

    env = env_factory()
    base = env.unwrapped
    base.config.render = RenderConfig(camera="side", **{field: (value, value)})
    expected = compute_angled_camera_params(
        spawn_center=base.config.spawn_center,
        spawn_max_radius=base.config.spawn_max_radius,
    )
    expected[key] = value
    viewer = Mock(cam=mujoco.MjvCamera(), lock=nullcontext)
    base._viewer = viewer
    env.reset(seed=0)
    for name in ("azimuth", "elevation", "distance"):
        assert getattr(viewer.cam, name) == pytest.approx(expected[name])
    np.testing.assert_allclose(viewer.cam.lookat, expected["lookat"])


@pytest.mark.parametrize("camera", ["side", "overhead"])
def test_disabling_ranges_restores_camera(env_factory, camera):
    env = env_factory(render_mode="rgb_array")
    base = env.unwrapped
    base.config.render = RenderConfig(
        width=64, height=48, camera="side", side_distance_range=(0.9, 0.9)
    )
    env.reset(seed=0)
    _render_or_skip(env)
    base.config.render = RenderConfig(width=64, height=48, camera=camera)
    env.reset(seed=0)
    _render_or_skip(env)
    params = base._render_camera_params()
    assert base._render_cam.distance == pytest.approx(params["distance"])
    assert base._render_cam.elevation == pytest.approx(params["elevation"])
