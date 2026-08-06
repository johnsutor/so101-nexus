"""YCB mesh geometry helpers for stable spawn poses in the MuJoCo backend."""

from __future__ import annotations

import numpy as np


def get_mujoco_ycb_rest_pose(verts: np.ndarray, margin: float = 0.002) -> tuple[np.ndarray, float]:
    """Return a stable object rest quaternion and spawn Z from mesh vertices.

    Objects are rotated so their thinnest axis points up (Z), producing a
    flat, stable rest pose. The quaternion uses the convention (w, x, y, z).
    ``verts`` are read in the body frame, so ``spawn_z`` is the height the body
    origin needs for the rotated mesh to clear the floor by ``margin``; vertices
    that are not centered on the origin are handled.
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
        # Thin axis is X - rotate 90 degrees around Y to bring X to Z, which maps
        # (x, y, z) to (z, y, -x). Negating any column other than the new Z would
        # invert the measured height, which cancels only for centered vertices.
        quat = np.array([_SQRT_HALF, 0.0, _SQRT_HALF, 0.0])
        rotated = verts[:, [2, 1, 0]].copy()
        rotated[:, 2] *= -1
        spawn_z = float(-np.min(rotated[:, 2])) + margin
    else:
        # Thin axis is Y - rotate 90 degrees around X to bring Y to Z, which maps
        # (x, y, z) to (x, -z, y).
        quat = np.array([_SQRT_HALF, _SQRT_HALF, 0.0, 0.0])
        rotated = verts[:, [0, 2, 1]].copy()
        rotated[:, 1] *= -1
        spawn_z = float(-np.min(rotated[:, 2])) + margin

    return quat, spawn_z
