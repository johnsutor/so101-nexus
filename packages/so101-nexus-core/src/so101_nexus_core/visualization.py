"""Shared visualization utilities for SO101-Nexus environments.

Provides camera view composition, frame grids, and video saving
that work across both MuJoCo and ManiSkill backends.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Sequence

import numpy as np

try:
    from PIL import Image, ImageDraw
    from PIL.Image import Resampling

    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

# Default tile dimensions for side-by-side view composition (4:3 aspect ratio).
DEFAULT_TILE_W = 320
DEFAULT_TILE_H = 240
# Height of the dark info bar appended below composed frames.
INFO_BAR_H = 28


@dataclass
class CameraView:
    """A named camera image."""

    name: str
    image: np.ndarray


def scalar(val: Any) -> float:
    """Extract a Python scalar from a tensor, array, or plain number."""
    if hasattr(val, "item"):
        return val.item()
    return float(np.asarray(val).squeeze())


def to_uint8(img: Any) -> np.ndarray:
    """Convert a camera image (tensor, float32 array, or uint8) to uint8 numpy."""
    if hasattr(img, "cpu"):
        img = img.cpu().numpy()
    img = np.asarray(img)
    if img.ndim == 4:
        img = img[0]
    if img.dtype != np.uint8:
        img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
    return img


def resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """Resize an image to (h, w) using Pillow if available, else nearest-neighbor."""
    if _PIL_AVAILABLE:
        return np.array(Image.fromarray(img).resize((w, h), Resampling.BILINEAR))
    ys = (np.arange(h) * img.shape[0] / h).astype(int)
    xs = (np.arange(w) * img.shape[1] / w).astype(int)
    return img[np.ix_(ys, xs)]


def add_label(img: np.ndarray, text: str) -> np.ndarray:
    """Overlay a small text label in the top-left corner of an image."""
    if not _PIL_AVAILABLE:
        return img
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    box_w = len(text) * 7 + 6
    draw.rectangle([0, 0, box_w, 15], fill=(0, 0, 0))
    draw.text((3, 2), text, fill=(255, 255, 255))
    return np.array(pil)


def make_info_bar(width: int, step: int, reward: float, success: bool) -> np.ndarray:
    """Create a dark info bar showing step, reward, and success status."""
    bar = np.full((INFO_BAR_H, width, 3), 20, dtype=np.uint8)
    if not _PIL_AVAILABLE:
        return bar
    pil = Image.fromarray(bar)
    draw = ImageDraw.Draw(pil)
    text = f"step: {step:>4d}   reward: {reward:+.4f}   success: {success}"
    color = (100, 220, 100) if success else (210, 210, 210)
    draw.text((8, (INFO_BAR_H - 11) // 2), text, fill=color)
    return np.array(pil)


def compose_frame(
    views: Sequence[CameraView],
    tile_w: int = DEFAULT_TILE_W,
    tile_h: int = DEFAULT_TILE_H,
    step: int | None = None,
    reward: float | None = None,
    success: bool | None = None,
) -> np.ndarray:
    """Tile camera views side-by-side with labels and an optional info bar.

    Parameters
    ----------
    views:
        One or more ``CameraView`` objects.
    tile_w, tile_h:
        Target width/height for each tile.
    step, reward, success:
        If all three are provided an info bar is appended below the tiles.

    Returns
    -------
    Composed uint8 RGB image as ``np.ndarray``.
    """
    tiles: list[np.ndarray] = []
    for v in views:
        img = to_uint8(v.image)
        img = resize(img, tile_w, tile_h)
        img = add_label(img, v.name)
        tiles.append(img)

    parts: list[np.ndarray] = []
    for i, t in enumerate(tiles):
        if i > 0:
            parts.append(np.full((tile_h, 2, 3), 60, dtype=np.uint8))
        parts.append(t)
    composed = np.concatenate(parts, axis=1)

    if step is not None and reward is not None and success is not None:
        bar = make_info_bar(composed.shape[1], step, reward, success)
        composed = np.concatenate([composed, bar], axis=0)

    return composed


def save_frame_grid(
    frames: Sequence[np.ndarray],
    path: str,
    cols: int = 4,
    max_frames: int = 16,
) -> None:
    """Save a grid of sampled frames as a PNG image."""
    import imageio.v3 as iio

    n_grid = min(max_frames, len(frames))
    indices = cast("list[int]", np.linspace(0, len(frames) - 1, n_grid, dtype=int).tolist())
    grid_frames = [frames[i] for i in indices]

    rows = (n_grid + cols - 1) // cols
    h, w, c = grid_frames[0].shape
    grid = np.zeros((rows * h, cols * w, c), dtype=np.uint8)
    for idx, f in enumerate(grid_frames):
        r, col_idx = divmod(idx, cols)
        grid[r * h : (r + 1) * h, col_idx * w : (col_idx + 1) * w] = f

    iio.imwrite(path, grid)


def save_video(
    frames: Sequence[np.ndarray],
    path: str,
    fps: int = 50,
) -> None:
    """Save a list of frames as an MP4 video."""
    import imageio.v3 as iio

    iio.imwrite(path, np.stack(frames), fps=fps)
