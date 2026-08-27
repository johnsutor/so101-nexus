"""YCB mesh geometry helpers for stable spawn poses in the MuJoCo backend."""

from __future__ import annotations

import numpy as np

# ``get_mujoco_ycb_rest_pose``'s AABB heuristic is a guess, not a physics
# result. ``scripts/validate_object_rest_poses.py`` settle-tests every
# supported model and records a corrected quaternion here when the
# heuristic pose is not settle-stable. It also runs a non-gating grasp
# screen (advisory only - see its module docstring for methodology and
# limits); ``GRASP_ADVISORY`` names objects whose narrowest sampled
# cross-section exceeds the gripper's full mechanical opening at every
# tested pose. Full results: src/so101_nexus/gso_pose_validation_results.json.
_TOO_WIDE = "no sampled slice fits the gripper's opening at any height/yaw; too wide to grasp."
GRASP_ADVISORY: dict[str, str] = {
    "037_scissors": _TOO_WIDE,
    "Shurtape_Gaffers_Tape_Silver_2_x_60_yd": _TOO_WIDE,
    "Big_O_Sponges_Assorted_Cellulose_12_pack": _TOO_WIDE,
    "Nestle_Raisinets_Milk_Chocolate_35_oz_992_g": _TOO_WIDE,
}

POSE_OVERRIDES: dict[str, tuple[float, float, float, float]] = {
    "030_fork": (
        0.9487688362649201,
        0.05192379329375049,
        0.015387636719621154,
        0.3112954154154558,
    ),
    "032_knife": (
        0.9370767109522925,
        0.3489106280209861,
        0.010219865920332777,
        -0.006645734376632088,
    ),
    "040_large_marker": (
        0.9868061963150316,
        -0.0036756068674924118,
        0.1618641914243856,
        6.605258629754342e-05,
    ),
    "043_phillips_screwdriver": (
        0.9869994957430818,
        0.02660164354705449,
        0.02117028873260648,
        0.1570864947679341,
    ),
    "Pony_C_Clamp_1440": (
        0.7252865768758933,
        -0.2984139667764899,
        0.36029176430034454,
        -0.5050725991515618,
    ),
    "OXO_Soft_Works_Can_Opener_SnapLock": (
        0.9124123190985273,
        -0.0022104422621626748,
        -0.005014702029669815,
        0.40923553934843887,
    ),
    "Shurtape_Gaffers_Tape_Silver_2_x_60_yd": (
        0.700240155426634,
        -0.01148416578855433,
        0.7137190536397818,
        -0.01170261234763929,
    ),
}


def _quat_to_rotmat(quat: np.ndarray) -> np.ndarray:
    """Return the 3x3 rotation matrix for a wxyz quaternion."""
    w, x, y, z = quat
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def get_mujoco_ycb_rest_pose(
    verts: np.ndarray, margin: float = 0.002, model_id: str | None = None
) -> tuple[np.ndarray, float]:
    """Return a stable object rest quaternion and spawn Z from mesh vertices.

    ``model_id`` looks up ``POSE_OVERRIDES`` first: a settle-test-validated
    correction for objects whose thin-axis-up heuristic proved unstable.
    Without a match (``model_id=None`` or not in the table), objects are
    rotated so their thinnest axis points up (Z), producing a flat, guessed
    rest pose. The quaternion uses the convention (w, x, y, z). ``verts`` are
    read in the body frame, so ``spawn_z`` is the height the body origin
    needs for the rotated mesh to clear the floor by ``margin``; vertices
    that are not centered on the origin are handled.
    """
    if model_id is not None and model_id in POSE_OVERRIDES:
        quat = np.array(POSE_OVERRIDES[model_id])
        rotated = verts @ _quat_to_rotmat(quat).T
        spawn_z = float(-np.min(rotated[:, 2])) + margin
        return quat, spawn_z

    extents = np.ptp(verts, axis=0)
    thin_axis = int(np.argmin(extents))

    # sqrt(2)/2 - the quaternion component for a 90-degree rotation.
    _SQRT_HALF = 0.7071068

    if thin_axis == 2:
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        spawn_z = float(-np.min(verts[:, 2])) + margin
    elif thin_axis == 0:
        # X up: rotate 90 deg around Y, mapping (x, y, z) -> (z, y, -x).
        quat = np.array([_SQRT_HALF, 0.0, _SQRT_HALF, 0.0])
        rotated = verts[:, [2, 1, 0]].copy()
        rotated[:, 2] *= -1
        spawn_z = float(-np.min(rotated[:, 2])) + margin
    else:
        # Y up: rotate 90 deg around X, mapping (x, y, z) -> (x, -z, y).
        quat = np.array([_SQRT_HALF, _SQRT_HALF, 0.0, 0.0])
        rotated = verts[:, [0, 2, 1]].copy()
        rotated[:, 1] *= -1
        spawn_z = float(-np.min(rotated[:, 2])) + margin

    return quat, spawn_z
