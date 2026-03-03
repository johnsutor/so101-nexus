"""Multi-camera visual tests for ManiSkill environments."""

from __future__ import annotations

import gymnasium
import numpy as np
import pytest

import so101_nexus_maniskill  # noqa: F401
from so101_nexus_core.visualization import CameraView, to_uint8

from .conftest import verify_scene

CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240

MANISKILL_SO101_ENVS = [
    "ManiSkillPickCubeGoalSO101-v1",
    "ManiSkillPickCubeLiftSO101-v1",
    "ManiSkillPickAndPlaceSO101-v1",
    "ManiSkillPickBananaGoalSO101-v1",
    "ManiSkillPickGolfBallGoalSO101-v1",
    "ManiSkillPickForkGoalSO101-v1",
]


def capture_maniskill_views(env: gymnasium.Env, obs: dict) -> list[CameraView]:
    """Capture base_camera, wrist_camera, and render camera views from a ManiSkill env."""
    base_img = to_uint8(obs["sensor_data"]["base_camera"]["rgb"])
    wrist_img = to_uint8(obs["sensor_data"]["wrist_camera"]["rgb"])

    render_img = to_uint8(env.render())

    return [
        CameraView(name="base_camera", image=base_img),
        CameraView(name="wrist_camera", image=wrist_img),
        CameraView(name="render_camera", image=render_img),
    ]


@pytest.mark.visual
class TestManiSkillCaptureInfrastructure:
    """Verify capture_maniskill_views returns valid images (no LLM needed)."""

    @pytest.mark.parametrize("env_id", MANISKILL_SO101_ENVS)
    def test_capture_views(self, env_id: str):
        _CUBE_ENVS = {
            "ManiSkillPickCubeGoalSO101-v1",
            "ManiSkillPickCubeLiftSO101-v1",
            "ManiSkillPickAndPlaceSO101-v1",
        }
        kwargs = dict(
            obs_mode="rgb",
            render_mode="rgb_array",
            camera_mode="both",
            camera_width=CAMERA_WIDTH,
            camera_height=CAMERA_HEIGHT,
            num_envs=1,
        )
        if env_id in _CUBE_ENVS:
            kwargs["cube_color"] = "red"
        env = gymnasium.make(env_id, **kwargs)
        obs, _ = env.reset(seed=42)
        views = capture_maniskill_views(env, obs)
        env.close()

        assert len(views) == 3
        expected_names = {"base_camera", "wrist_camera", "render_camera"}
        assert {v.name for v in views} == expected_names
        for v in views:
            assert v.image.dtype == np.uint8
            assert v.image.ndim == 3
            assert v.image.shape[2] == 3
            assert v.image.max() > 0, f"{v.name} is all-black"


ENV_DESCRIPTIONS = {
    "ManiSkillPickCubeGoalSO101-v1": "SO-101 robot arm pick-cube-goal task in ManiSkill",
    "ManiSkillPickCubeLiftSO101-v1": "SO-101 robot arm pick-cube-lift task in ManiSkill",
    "ManiSkillPickAndPlaceSO101-v1": (
        "SO-101 robot arm pick-and-place task in ManiSkill with a colored target disc"
    ),
    "ManiSkillPickBananaGoalSO101-v1": (
        "SO-101 robot arm picking up a banana (YCB object) in ManiSkill"
    ),
    "ManiSkillPickGolfBallGoalSO101-v1": (
        "SO-101 robot arm picking up a golf ball (YCB object) in ManiSkill"
    ),
    "ManiSkillPickForkGoalSO101-v1": (
        "SO-101 robot arm picking up a fork (YCB object) in ManiSkill"
    ),
}

EXPECTED_ELEMENTS = {
    "ManiSkillPickCubeGoalSO101-v1": "robot arm, red cube, ground plane, goal marker",
    "ManiSkillPickCubeLiftSO101-v1": "robot arm, red cube, ground plane",
    "ManiSkillPickAndPlaceSO101-v1": "robot arm, red cube, ground plane, colored target disc",
    "ManiSkillPickBananaGoalSO101-v1": (
        "robot arm, banana-shaped object, ground plane, goal marker"
    ),
    "ManiSkillPickGolfBallGoalSO101-v1": (
        "robot arm, small spherical object, ground plane, goal marker"
    ),
    "ManiSkillPickForkGoalSO101-v1": ("robot arm, fork-shaped object, ground plane, goal marker"),
}


@pytest.mark.visual
class TestManiSkillVisual:
    """LLM-powered visual verification of ManiSkill environments."""

    @pytest.mark.parametrize("env_id", MANISKILL_SO101_ENVS)
    def test_env_renders_correctly(self, visual_verifier, env_id: str):
        _CUBE_ENVS = {
            "ManiSkillPickCubeGoalSO101-v1",
            "ManiSkillPickCubeLiftSO101-v1",
            "ManiSkillPickAndPlaceSO101-v1",
        }
        kwargs = dict(
            obs_mode="rgb",
            render_mode="rgb_array",
            camera_mode="both",
            camera_width=CAMERA_WIDTH,
            camera_height=CAMERA_HEIGHT,
            num_envs=1,
        )
        if env_id in _CUBE_ENVS:
            kwargs["cube_color"] = "red"
        env = gymnasium.make(env_id, **kwargs)
        obs, _ = env.reset(seed=42)
        views = capture_maniskill_views(env, obs)
        env.close()

        passed, explanation = verify_scene(
            views=views,
            env_description=ENV_DESCRIPTIONS[env_id],
            expected_elements=EXPECTED_ELEMENTS[env_id],
            model=visual_verifier,
        )
        assert passed, f"Visual verification failed for {env_id}:\n{explanation}"
