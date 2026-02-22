"""Visualize the SO-101 ManiSkill pick-cube environment.

Renders both sensor cameras (base_camera | wrist_camera) tiled side by side
with a per-frame info bar showing step index, reward, and success status.

Saves:
  - episode_frames_maniskill.png  (4×N grid of sampled frames)
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

try:
    from PIL import Image, ImageDraw

    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

ENV_ID = "PickCubeLiftSO101-v1"
NUM_STEPS = 256

TILE_W = 320
TILE_H = 240

INFO_BAR_H = 28


def _to_uint8(img) -> np.ndarray:
    """Convert a ManiSkill camera image (tensor or ndarray) to uint8 numpy."""
    if hasattr(img, "cpu"):
        img = img.cpu().numpy()
    img = np.asarray(img)
    if img.ndim == 4:
        img = img[0]
    if img.dtype != np.uint8:
        img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
    return img


def _resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
    if _PIL_AVAILABLE:
        return np.array(Image.fromarray(img).resize((w, h), Image.BILINEAR))
    ys = (np.arange(h) * img.shape[0] / h).astype(int)
    xs = (np.arange(w) * img.shape[1] / w).astype(int)
    return img[np.ix_(ys, xs)]


def _add_label(img: np.ndarray, text: str) -> np.ndarray:
    if not _PIL_AVAILABLE:
        return img
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    box_w = len(text) * 7 + 6
    draw.rectangle([0, 0, box_w, 15], fill=(0, 0, 0))
    draw.text((3, 2), text, fill=(255, 255, 255))
    return np.array(pil)


def _make_info_bar(width: int, step: int, reward: float, success: bool) -> np.ndarray:
    bar = np.full((INFO_BAR_H, width, 3), 20, dtype=np.uint8)
    if not _PIL_AVAILABLE:
        return bar
    pil = Image.fromarray(bar)
    draw = ImageDraw.Draw(pil)
    text = f"step: {step:>4d}   reward: {reward:+.4f}   success: {success}"
    color = (100, 220, 100) if success else (210, 210, 210)
    draw.text((8, (INFO_BAR_H - 11) // 2), text, fill=color)
    return np.array(pil)


def _extract_camera(obs: dict, cam_name: str) -> np.ndarray:
    """Pull a uint8 RGB image from a ManiSkill observation dict."""
    return _to_uint8(obs["sensor_data"][cam_name]["rgb"])


def _scalar(val) -> float:
    """Extract a Python scalar from a tensor, array, or plain number."""
    if hasattr(val, "item"):
        return val.item()
    return float(np.asarray(val).squeeze())


def compose_frame(
    base_img: np.ndarray,
    wrist_img: np.ndarray,
    step: int,
    reward: float,
    success: bool,
) -> np.ndarray:
    """Tile base and wrist camera views side by side, then append info bar."""
    base = _add_label(_resize(base_img, TILE_W, TILE_H), "base_camera")
    wrist = _add_label(_resize(wrist_img, TILE_W, TILE_H), "wrist_camera")
    divider = np.full((TILE_H, 2, 3), 60, dtype=np.uint8)
    tiles = np.concatenate([base, divider, wrist], axis=1)
    bar = _make_info_bar(tiles.shape[1], step, reward, success)
    return np.concatenate([tiles, bar], axis=0)


def main():
    if not _PIL_AVAILABLE:
        print("Warning: Pillow not found. Text overlays will be skipped.")
        print("         Install with: pip install Pillow")

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
    base_img = _extract_camera(obs, "base_camera")
    wrist_img = _extract_camera(obs, "wrist_camera")
    frame = compose_frame(base_img, wrist_img, step=0, reward=0.0, success=False)
    frames.append(frame)

    print(f"Environment:          {ENV_ID}")
    print(f"Action space:         {env.action_space}")
    print(f"Composed frame shape: {frame.shape}  (H×W×C)")
    print(f"Running {NUM_STEPS} steps with random actions...")

    for step in range(1, NUM_STEPS + 1):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        base_img = _extract_camera(obs, "base_camera")
        wrist_img = _extract_camera(obs, "wrist_camera")
        r = _scalar(reward)
        s = bool(_scalar(info.get("success", False)))

        frame = compose_frame(base_img, wrist_img, step=step, reward=r, success=s)
        frames.append(frame)

        done = bool(_scalar(terminated)) or bool(_scalar(truncated))
        if done:
            print(f"Episode ended at step {step} (terminated={terminated})")
            break

    env.close()
    print(f"Collected {len(frames)} frames.")

    try:
        import imageio.v3 as iio

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

        iio.imwrite("episode_frames_maniskill.png", grid)
        print("Saved frame grid -> episode_frames_maniskill.png")
    except ImportError:
        print("Install imageio to save the frame grid: pip install imageio")

    try:
        import imageio.v3 as iio

        iio.imwrite("episode_maniskill.mp4", np.stack(frames), fps=50)
        print("Saved video       -> episode_maniskill.mp4")
    except ImportError:
        print("Install imageio[ffmpeg] to save the video: pip install imageio[ffmpeg]")
    except Exception as e:
        print(f"Could not save video (ffmpeg may be missing): {e}")


if __name__ == "__main__":
    main()
