"""Visualize the SO-101 MuJoCo pick-cube environment.

Renders a short episode with random actions and saves:
  - A grid of frames as a PNG image (episode_frames.png)
  - An MP4 video of the full episode (episode.mp4)

Usage:
    pip install -e packages/so101-nexus-core -e packages/so101-nexus-mujoco
    pip install imageio[ffmpeg]
    python visualize_env.py
"""

from __future__ import annotations

import gymnasium
import numpy as np

import so101_nexus_mujoco  # noqa: F401

ENV_ID = "MuJoCoPickCubeLift-v1"
NUM_STEPS = 256
RENDER_WIDTH = 640
RENDER_HEIGHT = 480


def main():
    env = gymnasium.make(
        ENV_ID,
        cube_color="red",
        camera_mode="wrist",
        camera_width=RENDER_WIDTH,
        camera_height=RENDER_HEIGHT,
    )
    obs, info = env.reset(seed=42)

    frames: list[np.ndarray] = []
    frame = obs["wrist_camera"]
    frames.append(frame)

    print(f"Environment: {ENV_ID}")
    print(f"Action space: {env.action_space}")
    print(f"Wrist camera frame shape: {frame.shape}")
    print(f"Running {NUM_STEPS} steps with random actions...")

    for step in range(NUM_STEPS):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        frame = obs["wrist_camera"]
        frames.append(frame)

        if terminated or truncated:
            print(f"Episode ended at step {step + 1} (terminated={terminated})")
            break

    env.close()
    print(f"Collected {len(frames)} frames.")

    # --- Save a grid of sampled frames as a PNG ---
    try:
        import imageio.v3 as iio

        # Pick up to 16 evenly spaced frames for the grid
        n_grid = min(16, len(frames))
        indices = np.linspace(0, len(frames) - 1, n_grid, dtype=int)
        grid_frames = [frames[i] for i in indices]

        cols = 4
        rows = (n_grid + cols - 1) // cols
        h, w, c = grid_frames[0].shape
        grid = np.zeros((rows * h, cols * w, c), dtype=np.uint8)
        for idx, f in enumerate(grid_frames):
            r, col = divmod(idx, cols)
            grid[r * h : (r + 1) * h, col * w : (col + 1) * w] = f

        iio.imwrite("episode_frames.png", grid)
        print("Saved frame grid  -> episode_frames.png")
    except ImportError:
        print("Install imageio to save the frame grid: pip install imageio")

    # --- Save an MP4 video ---
    try:
        import imageio.v3 as iio

        iio.imwrite("episode.mp4", np.stack(frames), fps=50)
        print("Saved video        -> episode.mp4")
    except ImportError:
        print("Install imageio[ffmpeg] to save the video: pip install imageio[ffmpeg]")
    except Exception as e:
        print(f"Could not save video (ffmpeg may be missing): {e}")


if __name__ == "__main__":
    main()
