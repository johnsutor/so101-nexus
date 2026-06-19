"""YCB mesh geometry helpers for stable spawn poses in the MuJoCo backend."""

from __future__ import annotations

import numpy as np


def get_mujoco_ycb_rest_pose(verts: np.ndarray, margin: float = 0.002) -> tuple[np.ndarray, float]:
    """Return a stable object rest quaternion and spawn Z from raw mesh vertices.

    Objects are rotated so their thinnest axis points up (Z), producing a
    flat, stable rest pose. The quaternion uses the convention (w, x, y, z).
    """
    extents = np.ptp(verts, axis=0)
    thin_axis = int(np.argmin(extents))

    # sqrt(2)/2 ≈ 0.7071068 - the quaternion component for a 90-degree rotation.
    _SQRT_HALF = 0.7071068

    if thin_axis == 2:
        # Thin axis is already Z - no rotation needed.
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        spawn_z = float(-np.min(verts[:, 2])) + margin
    elif thin_axis == 0:
        # Thin axis is X - rotate 90° around Y to bring X → Z.
        # Permute columns (X,Y,Z) → (Z,Y,X) then negate new-X to preserve handedness.
        quat = np.array([_SQRT_HALF, 0.0, _SQRT_HALF, 0.0])
        rotated = verts[:, [2, 1, 0]].copy()
        rotated[:, 0] *= -1
        spawn_z = float(-np.min(rotated[:, 2])) + margin
    else:
        # Thin axis is Y - rotate 90° around X to bring Y → Z.
        # Permute columns (X,Y,Z) → (X,Z,Y) then negate new-Y to preserve handedness.
        quat = np.array([_SQRT_HALF, _SQRT_HALF, 0.0, 0.0])
        rotated = verts[:, [0, 2, 1]].copy()
        rotated[:, 1] *= -1
        spawn_z = float(-np.min(rotated[:, 2])) + margin

    return quat, spawn_z
