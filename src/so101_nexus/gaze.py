"""Backend-shared gaze predicate primitives.

The MuJoCo and Warp backends both decide ``GazeState`` (and the look-at task's
success) from the angle between the wrist camera's optical axis and the
direction from that camera to the target object. The geometry lives here so the
two backends cannot drift, and so it is unit-testable without a simulator. Like
``so101_nexus.grasp``, these functions accept NumPy arrays and torch tensors by
duck typing; ``torch`` is never imported here.

The ray originates at the camera, not at the gripper tip: the wrist camera sits
about 8 cm from the TCP, which is the same order as the distance to a grasped
object, so a TCP-anchored ray answers a different question than "does the
camera see it".
"""

from __future__ import annotations

import numpy as np

# Degenerate-vector floor, matching so101_nexus.grasp's _EPS.
_EPS = 1e-9


def direction_to_object(camera_pos, object_pos):
    """Return the unit vector from the camera toward the object.

    Parameters
    ----------
    camera_pos : numpy.ndarray or torch.Tensor
        ``(..., 3)`` world-frame camera position.
    object_pos : numpy.ndarray or torch.Tensor
        ``(..., 3)`` world-frame target object position.

    Returns
    -------
    numpy.ndarray or torch.Tensor
        ``(..., 3)`` unit vector; a coincident camera and object give zeros.
    """
    v = object_pos - camera_pos
    if hasattr(v, "clamp"):  # torch.Tensor
        return v / v.norm(dim=-1, keepdim=True).clamp(min=_EPS)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), _EPS)


def gaze_cosine(gaze_axis, direction):
    """Return the cosine of the angle between the optical axis and *direction*.

    Both arguments are expected to be unit length (``gaze_axis`` comes straight
    out of a rotation matrix column, ``direction`` from
    :func:`direction_to_object`); the result is clamped to ``[-1, 1]`` so
    :func:`gaze_angle_rad` never sees an out-of-domain value from round-off.

    Parameters
    ----------
    gaze_axis : numpy.ndarray or torch.Tensor
        ``(..., 3)`` world-frame camera optical axis (where the camera points).
    direction : numpy.ndarray or torch.Tensor
        ``(..., 3)`` world-frame direction from the camera to the object.

    Returns
    -------
    numpy.floating or numpy.ndarray or torch.Tensor
        Cosine of the broadcast batch shape, in ``[-1, 1]``.
    """
    cos = (gaze_axis * direction).sum(-1)
    if hasattr(cos, "clamp"):  # torch.Tensor
        return cos.clamp(-1.0, 1.0)
    return np.clip(cos, -1.0, 1.0)


def gaze_angle_rad(cosine):
    """Return the gaze angle in radians for a cosine from :func:`gaze_cosine`.

    Parameters
    ----------
    cosine : float or numpy.ndarray or torch.Tensor
        Cosine in ``[-1, 1]``, as returned by :func:`gaze_cosine`.

    Returns
    -------
    numpy.floating or numpy.ndarray or torch.Tensor
        Angle in ``[0, pi]``. A Python float in gives a ``numpy.float64`` back.
    """
    if hasattr(cosine, "arccos"):  # torch.Tensor
        return cosine.arccos()
    return np.arccos(cosine)


def object_in_view(angle_rad, half_fov_rad):
    """Whether the object lies inside the camera's cone of half ``half_fov_rad``.

    A cone about the optical axis, so for a camera whose rendered image is wider
    than it is tall this is narrower than the visible frame (see
    :class:`so101_nexus.observations.GazeState`).

    Parameters
    ----------
    angle_rad : float or numpy.ndarray or torch.Tensor
        Gaze angle from :func:`gaze_angle_rad`.
    half_fov_rad : float or numpy.ndarray or torch.Tensor
        Half the camera's vertical field of view. An array or tensor value
        carries a per-world FOV, as wrist-camera domain randomization produces.

    Returns
    -------
    bool or numpy.bool_ or numpy.ndarray or torch.Tensor
        Boolean of the broadcast batch shape; two floats in give a plain ``bool``.
    """
    return angle_rad <= half_fov_rad
