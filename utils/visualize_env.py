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

from typing import TYPE_CHECKING, Protocol, cast

import gymnasium
import mujoco

if TYPE_CHECKING:
    import numpy as np

    from so101_nexus_core.config import EnvironmentConfig

import so101_nexus_mujoco  # noqa: F401
from so101_nexus_core.camera_utils import (
    compute_angled_camera_params,
    compute_overhead_camera_params,
)
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


class _MuJoCoEnvProtocol(Protocol):
    model: mujoco.MjModel
    data: mujoco.MjData
    config: EnvironmentConfig


def _capture_views(env: gymnasium.Env) -> list[CameraView]:
    """Capture wrist, top-down, and privileged camera views."""
    mujoco_env = cast("_MuJoCoEnvProtocol", env.unwrapped)
    model = mujoco_env.model
    data = mujoco_env.data
    config = mujoco_env.config

    wrist_cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
    renderer = mujoco.Renderer(model, height=RENDER_HEIGHT, width=RENDER_WIDTH)

    renderer.update_scene(data, camera=wrist_cam_id)
    wrist_img = renderer.render().copy()

    overhead_params = compute_overhead_camera_params(
        spawn_center=config.spawn_center,
        spawn_max_radius=config.spawn_max_radius,
    )
    top_cam = mujoco.MjvCamera()
    top_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    top_cam.lookat[:] = overhead_params["lookat"]
    top_cam.distance = overhead_params["distance"]
    top_cam.elevation = overhead_params["elevation"]
    top_cam.azimuth = overhead_params["azimuth"]
    renderer.update_scene(data, camera=top_cam)
    top_img = renderer.render().copy()

    angled_params = compute_angled_camera_params(
        spawn_center=config.spawn_center,
        spawn_max_radius=config.spawn_max_radius,
    )
    angled_cam = mujoco.MjvCamera()
    angled_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    angled_cam.lookat[:] = angled_params["lookat"]
    angled_cam.distance = angled_params["distance"]
    angled_cam.elevation = angled_params["elevation"]
    angled_cam.azimuth = angled_params["azimuth"]
    renderer.update_scene(data, camera=angled_cam)
    angled_img = renderer.render().copy()

    renderer.close()

    return [
        CameraView(name="wrist_camera", image=to_uint8(wrist_img)),
        CameraView(name="top_down", image=to_uint8(top_img)),
        CameraView(name="privileged", image=to_uint8(angled_img)),
    ]


def main():
    """Run a short rollout and save a tiled camera view to disk."""
    env = gymnasium.make(ENV_ID)
    obs, info = env.reset(seed=42)

    views = _capture_views(env)
    frame = compose_frame(views, tile_w=RENDER_WIDTH // 2, tile_h=RENDER_HEIGHT // 2)
    frames: list[np.ndarray] = [frame]

    print(f"Environment: {ENV_ID}")
    print(f"Action space: {env.action_space}")
    print(f"Composed frame shape: {frame.shape}  (HxWxC)")
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
