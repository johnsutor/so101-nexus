"""Tool-frame kinematics shared by the simulation backends.

Every function accepts NumPy arrays and torch tensors, dispatching by duck
typing so the scalar MuJoCo backend and the batched MuJoCo Warp backend call the
same code. ``torch`` is never imported here: a torch tensor can only exist if
the caller already imported torch, so the namespace is recovered from
``sys.modules``. All functions are shape-generic over leading batch dimensions.

Quaternions are ``[w, x, y, z]``, matching MuJoCo's convention and the
``EndEffectorPose`` observation. Orientations in *actions* are rotation vectors,
matching LeRobot's ``ee.wx``/``ee.wy``/``ee.wz`` feature naming.
"""

from __future__ import annotations

import sys

import numpy as np

# Damped-least-squares parameters. Three warm-started iterations track a 1 cm
# commanded step to a p90 error of 0.054 mm on the SO-101 model.
EE_IK_DAMPING = 0.05
EE_IK_ITERATIONS = 3

# The SO-101 arm has five actuated joints, so its tool Jacobian is rank 5 and one
# twist direction is always unreachable. Orientation is de-weighted rather than
# dropped, matching LeRobot's RobotKinematics.inverse_kinematics default, so
# position tracks essentially exactly and orientation is best-effort.
EE_ORIENTATION_WEIGHT = 0.01

# Physical scale of a +/-1 normalized pd_ee_delta_pose action: metres for the
# position triple, radians for the rotation-vector triple, and radians for the
# gripper (matching the joint-space _DELTA_ACTION_SCALE gripper entry).
EE_DELTA_ACTION_SCALE = (0.02, 0.02, 0.02, 0.1, 0.1, 0.1, 0.2)

# [x, y, z, wx, wy, wz, gripper].
EE_ACTION_DIM = 7

_SMALL_ANGLE = 1e-8


def _namespace(x):
    """Return the array namespace backing ``x`` (NumPy or the imported torch)."""
    if hasattr(x, "clamp"):  # torch.Tensor
        return sys.modules["torch"]
    return np


def _eye_like(x, n: int):
    """Return an ``(n, n)`` identity matching ``x``'s dtype, device, and library."""
    xp = _namespace(x)
    if xp is np:
        return np.eye(n, dtype=x.dtype)
    return xp.eye(n, dtype=x.dtype, device=x.device)


def quat_conjugate(quat):
    """Conjugate of a ``[w, x, y, z]`` quaternion, shape ``(..., 4)``."""
    xp = _namespace(quat)
    return xp.concatenate([quat[..., :1], -quat[..., 1:]], axis=-1)


def quat_multiply(a, b):
    """Hamilton product ``a * b`` of ``[w, x, y, z]`` quaternions, shape ``(..., 4)``."""
    xp = _namespace(a)
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return xp.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def quat_to_rotvec(quat):
    """Convert ``[w, x, y, z]`` quaternions to rotation vectors, shape ``(..., 3)``.

    The returned rotation has an angle in ``[0, pi]``: the quaternion double cover
    is resolved by taking the shortest-path sign, so ``q`` and ``-q`` agree.

    Parameters
    ----------
    quat : numpy.ndarray or torch.Tensor
        Unit quaternions of shape ``(..., 4)``.
    """
    xp = _namespace(quat)
    # Shortest path: flip so the scalar part is non-negative.
    quat = xp.where(quat[..., :1] < 0.0, -quat, quat)
    w = quat[..., :1]
    vec = quat[..., 1:]
    vec_norm = xp.linalg.norm(vec, axis=-1, keepdims=True)
    angle = 2.0 * xp.arctan2(vec_norm, w)
    # angle / |v| -> 2 / w as |v| -> 0, and w -> 1 there.
    scale = xp.where(vec_norm > _SMALL_ANGLE, angle / xp.clip(vec_norm, _SMALL_ANGLE, None), 2.0)
    return vec * scale


def rotvec_to_quat(rotvec):
    """Convert rotation vectors to ``[w, x, y, z]`` quaternions, shape ``(..., 4)``.

    Parameters
    ----------
    rotvec : numpy.ndarray or torch.Tensor
        Rotation vectors of shape ``(..., 3)``; the norm is the angle in radians.
    """
    xp = _namespace(rotvec)
    angle = xp.linalg.norm(rotvec, axis=-1, keepdims=True)
    half = 0.5 * angle
    # sin(a/2) / a -> 1/2 as a -> 0.
    scale = xp.where(angle > _SMALL_ANGLE, xp.sin(half) / xp.clip(angle, _SMALL_ANGLE, None), 0.5)
    return xp.concatenate([xp.cos(half), rotvec * scale], axis=-1)


def orientation_error(current_quat, target_quat):
    """World-frame rotation vector taking ``current_quat`` onto ``target_quat``.

    The result pairs with a world-frame rotational Jacobian (``mj_jacSite``'s
    ``jacr``), so the error is the left-multiplied relative rotation.

    Parameters
    ----------
    current_quat, target_quat : numpy.ndarray or torch.Tensor
        Unit ``[w, x, y, z]`` quaternions of shape ``(..., 4)``.
    """
    return quat_to_rotvec(quat_multiply(target_quat, quat_conjugate(current_quat)))


def pose_error(
    current_pos,
    current_quat,
    target_pos,
    target_quat,
    *,
    orientation_weight: float = EE_ORIENTATION_WEIGHT,
):
    """Weighted ``(..., 6)`` twist error ``[dx, dy, dz, dwx, dwy, dwz]``.

    Parameters
    ----------
    current_pos, target_pos : numpy.ndarray or torch.Tensor
        Tool positions of shape ``(..., 3)`` in world metres.
    current_quat, target_quat : numpy.ndarray or torch.Tensor
        Tool orientations of shape ``(..., 4)`` as ``[w, x, y, z]`` quaternions.
    orientation_weight : float
        Relative weight of the rotational error. The default de-weights
        orientation because the five-joint arm cannot realize arbitrary poses.
    """
    xp = _namespace(current_pos)
    rot_err = orientation_error(current_quat, target_quat) * orientation_weight
    return xp.concatenate([target_pos - current_pos, rot_err], axis=-1)


def damped_least_squares(jac, error, *, damping: float = EE_IK_DAMPING):
    """Solve ``jac @ dq ~= error`` with Levenberg-Marquardt damping.

    Returns ``jac.T @ inv(jac @ jac.T + damping^2 I) @ error``, which stays finite
    through the rank-deficient configurations the five-joint SO-101 arm is always
    in.

    Parameters
    ----------
    jac : numpy.ndarray or torch.Tensor
        Tool Jacobian of shape ``(..., 6, n_joints)``.
    error : numpy.ndarray or torch.Tensor
        Twist error of shape ``(..., 6)``.
    damping : float
        Damping factor; larger values trade tracking accuracy for conditioning.
    """
    xp = _namespace(jac)
    jac_t = jac.swapaxes(-1, -2)
    gram = jac @ jac_t + (damping * damping) * _eye_like(jac, jac.shape[-2])
    if xp is np:
        solution = np.linalg.solve(gram, error[..., None])
    else:
        # solve_ex skips the singularity check that torch.linalg.solve performs,
        # which forces a device-to-host sync every call. Damping keeps the Gram
        # matrix positive definite, so the check has nothing to find and the sync
        # costs more than the whole solve on a batched GPU rollout.
        solution = xp.linalg.solve_ex(gram, error[..., None])[0]
    return (jac_t @ solution)[..., 0]


def ee_ik_delta_q(
    jac,
    current_pos,
    current_quat,
    target_pos,
    target_quat,
    *,
    orientation_weight: float = EE_ORIENTATION_WEIGHT,
    damping: float = EE_IK_DAMPING,
):
    """One damped-least-squares joint increment toward a target tool pose.

    Backends iterate this, re-evaluating forward kinematics between calls;
    the loop lives in the backend because that re-evaluation is backend-specific.
    Pass the raw tool Jacobian: this function applies ``orientation_weight`` to its
    rotational rows itself, so no backend can pair a weighted error with an
    unweighted Jacobian.

    Parameters
    ----------
    jac : numpy.ndarray or torch.Tensor
        Raw, unweighted tool Jacobian of shape ``(..., 6, n_joints)``: three
        translational rows then three rotational rows, as ``mj_jacSite`` produces.
    current_pos, target_pos : numpy.ndarray or torch.Tensor
        Tool positions of shape ``(..., 3)``.
    current_quat, target_quat : numpy.ndarray or torch.Tensor
        Tool orientations of shape ``(..., 4)`` as ``[w, x, y, z]`` quaternions.
    orientation_weight : float
        Relative weight of the rotational rows.
    damping : float
        Damping factor.
    """
    xp = _namespace(jac)
    error = pose_error(
        current_pos,
        current_quat,
        target_pos,
        target_quat,
        orientation_weight=orientation_weight,
    )
    # Weighted least squares scales both sides. De-weighting only the rotational
    # error would leave the solver holding tool orientation at full weight, which
    # a five-joint arm can satisfy only by barely moving, collapsing position
    # tracking (measured: 4.63 mm median error against 0.16 mm when both are scaled).
    weighted_jac = xp.concatenate([jac[..., :3, :], jac[..., 3:, :] * orientation_weight], axis=-2)
    return damped_least_squares(weighted_jac, error, damping=damping)
