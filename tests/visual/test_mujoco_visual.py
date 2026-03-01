"""Multi-camera visual tests for MuJoCo environments."""

from __future__ import annotations

import gymnasium
import mujoco
import numpy as np
import pytest

import so101_nexus_mujoco  # noqa: F401
from so101_nexus_core.visualization import CameraView, to_uint8

from .conftest import verify_scene

CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240

WORKSPACE_CENTER = np.array([0.15, 0.0, 0.0])

MUJOCO_ENVS = [
    "MuJoCoPickCubeGoal-v1",
    "MuJoCoPickCubeLift-v1",
    "MuJoCoPickAndPlace-v1",
]


def capture_mujoco_views(env: gymnasium.Env) -> list[CameraView]:
    """Capture wrist, top-down, and head-on camera views from a MuJoCo env."""
    wrist_cam_id = mujoco.mj_name2id(env.unwrapped.model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
    renderer = mujoco.Renderer(env.unwrapped.model, height=CAMERA_HEIGHT, width=CAMERA_WIDTH)

    renderer.update_scene(env.unwrapped.data, camera=wrist_cam_id)
    wrist_img = renderer.render().copy()

    top_cam = mujoco.MjvCamera()
    top_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    top_cam.lookat[:] = WORKSPACE_CENTER
    top_cam.distance = 0.6
    top_cam.elevation = -90
    top_cam.azimuth = 90
    renderer.update_scene(env.unwrapped.data, camera=top_cam)
    top_img = renderer.render().copy()

    head_cam = mujoco.MjvCamera()
    head_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    head_cam.lookat[:] = WORKSPACE_CENTER
    head_cam.distance = 0.6
    head_cam.elevation = -15
    head_cam.azimuth = 180
    renderer.update_scene(env.unwrapped.data, camera=head_cam)
    head_img = renderer.render().copy()

    renderer.close()

    return [
        CameraView(name="wrist_camera", image=to_uint8(wrist_img)),
        CameraView(name="top_down", image=to_uint8(top_img)),
        CameraView(name="head_on", image=to_uint8(head_img)),
    ]


@pytest.mark.visual
class TestMuJoCoCaptureInfrastructure:
    """Verify capture_mujoco_views returns valid images (no LLM needed)."""

    @pytest.mark.parametrize("env_id", MUJOCO_ENVS)
    def test_capture_views(self, env_id: str):
        env = gymnasium.make(
            env_id,
            camera_mode="wrist",
            camera_width=CAMERA_WIDTH,
            camera_height=CAMERA_HEIGHT,
        )
        env.reset(seed=42)
        views = capture_mujoco_views(env)
        env.close()

        assert len(views) == 3
        expected_names = {"wrist_camera", "top_down", "head_on"}
        assert {v.name for v in views} == expected_names
        for v in views:
            assert v.image.dtype == np.uint8
            assert v.image.shape == (CAMERA_HEIGHT, CAMERA_WIDTH, 3)
            assert v.image.max() > 0, f"{v.name} is all-black"


ENV_DESCRIPTIONS = {
    "MuJoCoPickCubeGoal-v1": "SO-101 robot arm pick-cube-goal task in MuJoCo",
    "MuJoCoPickCubeLift-v1": "SO-101 robot arm pick-cube-lift task in MuJoCo",
    "MuJoCoPickAndPlace-v1": (
        "SO-101 robot arm pick-and-place task in MuJoCo with a colored target disc"
    ),
}

EXPECTED_ELEMENTS = {
    "MuJoCoPickCubeGoal-v1": "robot arm, red cube, ground plane, goal marker",
    "MuJoCoPickCubeLift-v1": "robot arm, red cube, ground plane",
    "MuJoCoPickAndPlace-v1": "robot arm, red cube, ground plane, colored target disc",
}


@pytest.mark.visual
class TestMuJoCoVisual:
    """LLM-powered visual verification of MuJoCo environments."""

    @pytest.mark.parametrize("env_id", MUJOCO_ENVS)
    def test_env_renders_correctly(self, visual_verifier, env_id: str):
        env = gymnasium.make(
            env_id,
            cube_color="red",
            camera_mode="wrist",
            camera_width=CAMERA_WIDTH,
            camera_height=CAMERA_HEIGHT,
        )
        env.reset(seed=42)
        views = capture_mujoco_views(env)
        env.close()

        passed, explanation = verify_scene(
            views=views,
            env_description=ENV_DESCRIPTIONS[env_id],
            expected_elements=EXPECTED_ELEMENTS[env_id],
            model=visual_verifier,
        )
        assert passed, f"Visual verification failed for {env_id}:\n{explanation}"
