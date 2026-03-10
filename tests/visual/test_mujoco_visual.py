"""Multi-camera visual tests for MuJoCo environments."""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import so101_nexus_mujoco  # noqa: F401, E402
from so101_nexus_core.config import (  # noqa: E402
    CameraConfig,
    PickAndPlaceConfig,
    PickCubeConfig,
    PickYCBConfig,
)
from so101_nexus_core.visualization import CameraView, to_uint8  # noqa: E402

from .conftest import verify_scene  # noqa: E402

CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240
_CAM = CameraConfig(width=CAMERA_WIDTH, height=CAMERA_HEIGHT)

WORKSPACE_CENTER = np.array([0.15, 0.0, 0.0])

_PICK_CUBE_ENVS = {
    "MuJoCoPickCubeLift-v1",
}
_PICK_AND_PLACE_ENVS = {
    "MuJoCoPickAndPlace-v1",
}


def _env_kwargs(env_id: str) -> dict:
    """Return config kwargs appropriate for *env_id*."""
    if env_id in _PICK_CUBE_ENVS:
        return dict(config=PickCubeConfig(camera=_CAM, cube_colors="red"))
    if env_id in _PICK_AND_PLACE_ENVS:
        return dict(config=PickAndPlaceConfig(camera=_CAM, cube_colors="red"))
    return dict(config=PickYCBConfig(camera=_CAM))


MUJOCO_ENVS = [
    "MuJoCoPickCubeLift-v1",
    "MuJoCoPickAndPlace-v1",
    "MuJoCoPickBananaLift-v1",
    "MuJoCoPickGolfBallLift-v1",
    "MuJoCoPickForkLift-v1",
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
            **_env_kwargs(env_id),
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
    "MuJoCoPickCubeLift-v1": "SO-101 robot arm pick-cube-lift task in MuJoCo",
    "MuJoCoPickAndPlace-v1": (
        "SO-101 robot arm pick-and-place task in MuJoCo with a colored target disc"
    ),
    "MuJoCoPickBananaLift-v1": "SO-101 robot arm picking up a banana (YCB object) in MuJoCo",
    "MuJoCoPickGolfBallLift-v1": "SO-101 robot arm picking up a golf ball (YCB object) in MuJoCo",
    "MuJoCoPickForkLift-v1": "SO-101 robot arm picking up a fork (YCB object) in MuJoCo",
}

EXPECTED_ELEMENTS = {
    "MuJoCoPickCubeLift-v1": "robot arm, red cube, ground plane",
    "MuJoCoPickAndPlace-v1": "robot arm, red cube, ground plane, colored target disc",
    "MuJoCoPickBananaLift-v1": "robot arm, banana-shaped object, ground plane",
    "MuJoCoPickGolfBallLift-v1": "robot arm, small spherical object, ground plane",
    "MuJoCoPickForkLift-v1": "robot arm, fork-shaped object, ground plane",
}


@pytest.mark.visual
class TestMuJoCoVisual:
    """LLM-powered visual verification of MuJoCo environments."""

    @pytest.mark.parametrize("env_id", MUJOCO_ENVS)
    def test_env_renders_correctly(self, visual_verifier, env_id: str):
        env = gymnasium.make(
            env_id,
            camera_mode="wrist",
            **_env_kwargs(env_id),
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
