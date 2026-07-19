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
