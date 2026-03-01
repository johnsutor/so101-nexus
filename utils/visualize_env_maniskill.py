"""Visualize the SO-101 ManiSkill pick-cube environment.

Renders both sensor cameras (base_camera | wrist_camera) tiled side by side
with a per-frame info bar showing step index, reward, and success status.

Saves:
  - episode_frames_maniskill.png  (4xN grid of sampled frames)
  - episode_maniskill.mp4         (full episode video at 50 fps)

Usage:
    pip install -e packages/so101-nexus-core -e packages/so101-nexus-maniskill
    pip install imageio[ffmpeg] Pillow
    python visualize_env_maniskill.py
"""

from __future__ import annotations

import gymnasium
import numpy as np

import so101_nexus_maniskill  # noqa: F401
from so101_nexus_core.visualization import (
    CameraView,
    compose_frame,
    save_frame_grid,
    save_video,
    scalar,
    to_uint8,
)

ENV_ID = "ManiSkillPickAndPlaceSO101-v1"
NUM_STEPS = 256

TILE_W = 320
TILE_H = 240


def _extract_views(obs: dict) -> list[CameraView]:
    """Pull base_camera and wrist_camera views from a ManiSkill observation."""
    return [
        CameraView(name="base_camera", image=to_uint8(obs["sensor_data"]["base_camera"]["rgb"])),
        CameraView(name="wrist_camera", image=to_uint8(obs["sensor_data"]["wrist_camera"]["rgb"])),
    ]


def main():
    env = gymnasium.make(
        ENV_ID,
        obs_mode="rgb",
        render_mode="rgb_array",
        camera_mode="both",
        cube_color="red",
        camera_width=TILE_W,
        camera_height=TILE_H,
        num_envs=1,
    )
    obs, info = env.reset(seed=42)

    frames: list[np.ndarray] = []
    views = _extract_views(obs)
    frame = compose_frame(views, tile_w=TILE_W, tile_h=TILE_H, step=0, reward=0.0, success=False)
    frames.append(frame)

    print(f"Environment:          {ENV_ID}")
    print(f"Action space:         {env.action_space}")
    print(f"Composed frame shape: {frame.shape}  (H×W×C)")
    print(f"Running {NUM_STEPS} steps with random actions...")

    for step in range(1, NUM_STEPS + 1):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        views = _extract_views(obs)
        r = scalar(reward)
        s = bool(scalar(info.get("success", False)))

        frame = compose_frame(views, tile_w=TILE_W, tile_h=TILE_H, step=step, reward=r, success=s)
        frames.append(frame)

        done = bool(scalar(terminated)) or bool(scalar(truncated))
        if done:
            print(f"Episode ended at step {step} (terminated={terminated})")
            break

    env.close()
    print(f"Collected {len(frames)} frames.")

    try:
        save_frame_grid(frames, "episode_frames_maniskill.png")
        print("Saved frame grid -> episode_frames_maniskill.png")
    except ImportError:
        print("Install imageio to save the frame grid: pip install imageio")

    try:
        save_video(frames, "episode_maniskill.mp4")
        print("Saved video       -> episode_maniskill.mp4")
    except ImportError:
        print("Install imageio[ffmpeg] to save the video: pip install imageio[ffmpeg]")
    except Exception as e:
        print(f"Could not save video (ffmpeg may be missing): {e}")


if __name__ == "__main__":
    main()
