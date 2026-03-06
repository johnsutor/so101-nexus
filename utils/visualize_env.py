"""Visualize the SO-101 MuJoCo pick-cube environment.

Renders a short episode with random actions and saves:
  - A grid of frames as a PNG image (episode_frames.png)
  - An MP4 video of the full episode (episode.mp4)

Usage:
    pip install -e packages/so101-nexus-core -e packages/so101-nexus-mujoco
    pip install imageio[ffmpeg] Pillow
    python visualize_env.py
"""

from __future__ import annotations

from typing import Protocol, cast

import gymnasium
import mujoco
import numpy as np

import so101_nexus_mujoco  # noqa: F401
from so101_nexus_core.visualization import (
    CameraView,
    compose_frame,
    save_frame_grid,
    save_video,
    to_uint8,
)

ENV_ID = "MuJoCoPickAndPlace-v1"
NUM_STEPS = 256
RENDER_WIDTH = 640
RENDER_HEIGHT = 480

WORKSPACE_CENTER = np.array([0.15, 0.0, 0.0])


class _MuJoCoEnvProtocol(Protocol):
    model: mujoco.MjModel
    data: mujoco.MjData


def _capture_views(env: gymnasium.Env) -> list[CameraView]:
    """Capture wrist, top-down, and head-on camera views."""
    mujoco_env = cast(_MuJoCoEnvProtocol, env.unwrapped)
    model = mujoco_env.model
    data = mujoco_env.data

    wrist_cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
    renderer = mujoco.Renderer(model, height=RENDER_HEIGHT, width=RENDER_WIDTH)

    renderer.update_scene(data, camera=wrist_cam_id)
    wrist_img = renderer.render().copy()

    top_cam = mujoco.MjvCamera()
    top_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    top_cam.lookat[:] = WORKSPACE_CENTER
    top_cam.distance = 0.6
    top_cam.elevation = -90
    top_cam.azimuth = 90
    renderer.update_scene(data, camera=top_cam)
    top_img = renderer.render().copy()

    head_cam = mujoco.MjvCamera()
    head_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    head_cam.lookat[:] = WORKSPACE_CENTER
    head_cam.distance = 0.6
    head_cam.elevation = -15
    head_cam.azimuth = 180
    renderer.update_scene(data, camera=head_cam)
    head_img = renderer.render().copy()

    renderer.close()

    return [
        CameraView(name="wrist_camera", image=to_uint8(wrist_img)),
        CameraView(name="top_down", image=to_uint8(top_img)),
        CameraView(name="head_on", image=to_uint8(head_img)),
    ]


def main():
    env = gymnasium.make(
        ENV_ID,
        cube_color="red",
        camera_mode="wrist",
        camera_width=RENDER_WIDTH,
        camera_height=RENDER_HEIGHT,
    )
    obs, info = env.reset(seed=42)

    views = _capture_views(env)
    frame = compose_frame(views, tile_w=RENDER_WIDTH // 2, tile_h=RENDER_HEIGHT // 2)
    frames: list[np.ndarray] = [frame]

    print(f"Environment: {ENV_ID}")
    print(f"Action space: {env.action_space}")
    print(f"Composed frame shape: {frame.shape}  (H×W×C)")
    print(f"Running {NUM_STEPS} steps with random actions...")

    for step in range(NUM_STEPS):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        views = _capture_views(env)
        frame = compose_frame(
            views,
            tile_w=RENDER_WIDTH // 2,
            tile_h=RENDER_HEIGHT // 2,
            step=step + 1,
            reward=float(reward),
            success=bool(info.get("success", False)),
        )
        frames.append(frame)

        if terminated or truncated:
            print(f"Episode ended at step {step + 1} (terminated={terminated})")
            break

    env.close()
    print(f"Collected {len(frames)} frames.")

    try:
        save_frame_grid(frames, "episode_frames.png")
        print("Saved frame grid  -> episode_frames.png")
    except ImportError:
        print("Install imageio to save the frame grid: pip install imageio")

    try:
        save_video(frames, "episode.mp4")
        print("Saved video        -> episode.mp4")
    except ImportError:
        print("Install imageio[ffmpeg] to save the video: pip install imageio[ffmpeg]")
    except Exception as e:
        print(f"Could not save video (ffmpeg may be missing): {e}")


if __name__ == "__main__":
    main()
