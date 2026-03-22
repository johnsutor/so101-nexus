"""Multi-camera visual tests for ManiSkill environments."""

from __future__ import annotations

import gymnasium
import numpy as np
import pytest

import so101_nexus_maniskill  # noqa: F401
from so101_nexus_core.config import (
    PickAndPlaceConfig,
    PickConfig,
    RenderConfig,
)
from so101_nexus_core.objects import CubeObject
from so101_nexus_core.observations import JointPositions, OverheadCamera, WristCamera
from so101_nexus_core.visualization import CameraView, to_uint8

from .conftest import verify_scene

CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240
_RENDER = RenderConfig(width=CAMERA_WIDTH, height=CAMERA_HEIGHT)

_PICK_LIFT_ENVS = {
    "ManiSkillPickLiftSO101-v1",
}
_PICK_AND_PLACE_ENVS = {
    "ManiSkillPickAndPlaceSO101-v1",
}


def _env_kwargs(env_id: str) -> dict:
    """Return config kwargs appropriate for *env_id*."""
    if env_id in _PICK_LIFT_ENVS:
        return {
            "config": PickConfig(
                objects=[CubeObject(color="red")],
                render=_RENDER,
                observations=[
                    JointPositions(),
                    WristCamera(width=CAMERA_WIDTH, height=CAMERA_HEIGHT),
                    OverheadCamera(width=CAMERA_WIDTH, height=CAMERA_HEIGHT),
                ],
            )
        }
    if env_id in _PICK_AND_PLACE_ENVS:
        return {
            "config": PickAndPlaceConfig(
                render=_RENDER,
                cube_colors="red",
                observations=[
                    JointPositions(),
                    WristCamera(width=CAMERA_WIDTH, height=CAMERA_HEIGHT),
                    OverheadCamera(width=CAMERA_WIDTH, height=CAMERA_HEIGHT),
                ],
            )
        }
    return {
        "config": PickConfig(
            render=_RENDER,
            observations=[
                JointPositions(),
                WristCamera(width=CAMERA_WIDTH, height=CAMERA_HEIGHT),
                OverheadCamera(width=CAMERA_WIDTH, height=CAMERA_HEIGHT),
            ],
        )
    }


MANISKILL_SO101_ENVS = [
    "ManiSkillPickLiftSO101-v1",
    "ManiSkillPickAndPlaceSO101-v1",
]


def capture_maniskill_views(env: gymnasium.Env, obs: dict) -> list[CameraView]:
    """Capture overhead_camera, wrist_camera, and render camera views from a ManiSkill env."""
    overhead_img = to_uint8(obs["sensor_data"]["overhead_camera"]["rgb"])
    wrist_img = to_uint8(obs["sensor_data"]["wrist_camera"]["rgb"])

    render_img = to_uint8(env.render())

    return [
        CameraView(name="overhead_camera", image=overhead_img),
        CameraView(name="wrist_camera", image=wrist_img),
        CameraView(name="render_camera", image=render_img),
    ]


@pytest.mark.visual
class TestManiSkillCaptureInfrastructure:
    """Verify capture_maniskill_views returns valid images (no LLM needed)."""

    @pytest.mark.parametrize("env_id", MANISKILL_SO101_ENVS)
    def test_capture_views(self, env_id: str):
        kwargs = dict(
            obs_mode="rgb",
            render_mode="rgb_array",
            num_envs=1,
            **_env_kwargs(env_id),
        )
        env = gymnasium.make(env_id, **kwargs)
        obs, _ = env.reset(seed=42)
        views = capture_maniskill_views(env, obs)
        env.close()

        assert len(views) == 3
        expected_names = {"overhead_camera", "wrist_camera", "render_camera"}
        assert {v.name for v in views} == expected_names
        for v in views:
            assert v.image.dtype == np.uint8
            assert v.image.ndim == 3
            assert v.image.shape[2] == 3
            assert v.image.max() > 0, f"{v.name} is all-black"


ENV_DESCRIPTIONS = {
    "ManiSkillPickLiftSO101-v1": "SO-101 robot arm pick-and-lift task in ManiSkill",
    "ManiSkillPickAndPlaceSO101-v1": (
        "SO-101 robot arm pick-and-place task in ManiSkill with a colored target disc"
    ),
}

EXPECTED_ELEMENTS = {
    "ManiSkillPickLiftSO101-v1": "robot arm, red cube, ground plane",
    "ManiSkillPickAndPlaceSO101-v1": "robot arm, red cube, ground plane, colored target disc",
}


@pytest.mark.visual
class TestManiSkillVisual:
    """LLM-powered visual verification of ManiSkill environments."""

    @pytest.mark.parametrize("env_id", MANISKILL_SO101_ENVS)
    def test_env_renders_correctly(self, visual_verifier, env_id: str):
        kwargs = dict(
            obs_mode="rgb",
            render_mode="rgb_array",
            num_envs=1,
            **_env_kwargs(env_id),
        )
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
