"""YCB mesh geometry helpers for stable spawn poses across simulation backends."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np


@lru_cache(maxsize=1)
def _load_maniskill_pick_db() -> dict:
    info_path = (
        Path.home() / ".maniskill" / "data" / "assets" / "mani_skill2_ycb" / "info_pick_v0.json"
    )
    with open(info_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_maniskill_ycb_spawn_z(model_id: str, margin: float = 0.002) -> float:
    """Compute stable spawn Z for ManiSkill YCB placement from metadata bounds."""
    model_db = _load_maniskill_pick_db()
    metadata = model_db[model_id]
    scale = metadata.get("scales", [1.0])[0]
    bbox_min_z = metadata["bbox"]["min"][2] * scale
    return float(-bbox_min_z + margin)


def get_mujoco_ycb_rest_pose(verts: np.ndarray, margin: float = 0.002) -> tuple[np.ndarray, float]:
    """Return a stable object rest quaternion and spawn Z from raw mesh vertices.

    Objects are rotated so their thinnest axis points up (Z), producing a
    flat, stable rest pose. The quaternion uses the convention (w, x, y, z).
    """
    extents = np.ptp(verts, axis=0)
    thin_axis = int(np.argmin(extents))

    # sqrt(2)/2 ≈ 0.7071068 — the quaternion component for a 90-degree rotation.
    _SQRT_HALF = 0.7071068

    if thin_axis == 2:
        # Thin axis is already Z — no rotation needed.
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        spawn_z = float(-np.min(verts[:, 2])) + margin
    elif thin_axis == 0:
        # Thin axis is X — rotate 90° around Y to bring X → Z.
        # Permute columns (X,Y,Z) → (Z,Y,X) then negate new-X to preserve handedness.
        quat = np.array([_SQRT_HALF, 0.0, _SQRT_HALF, 0.0])
        rotated = verts[:, [2, 1, 0]].copy()
        rotated[:, 0] *= -1
        spawn_z = float(-np.min(rotated[:, 2])) + margin
    else:
        # Thin axis is Y — rotate 90° around X to bring Y → Z.
        # Permute columns (X,Y,Z) → (X,Z,Y) then negate new-Y to preserve handedness.
        quat = np.array([_SQRT_HALF, _SQRT_HALF, 0.0, 0.0])
        rotated = verts[:, [0, 2, 1]].copy()
        rotated[:, 1] *= -1
        spawn_z = float(-np.min(rotated[:, 2])) + margin

    return quat, spawn_z
