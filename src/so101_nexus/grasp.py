"""Backend-shared grasp predicate primitives.

The MuJoCo and Warp backends both decide ``GraspState`` from contacts between
the target object and the two finger geom sets. The geometric part of that
decision lives here so the two backends cannot drift, and so it is unit-testable
without a simulator. Like ``so101_nexus.rewards``, these functions accept NumPy
arrays and torch tensors by duck typing; ``torch`` is never imported here.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-9


def _normalize_last(v):
    """Unit-normalize along the last axis, leaving near-zero vectors at zero."""
    if hasattr(v, "clamp"):  # torch.Tensor
        return v / v.norm(dim=-1, keepdim=True).clamp(min=_EPS)
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(norm, _EPS)


def opposing_normals_ok(gripper_normal, jaw_normal, *, threshold):
    """Return whether the two finger sets push the object from opposing sides.

    Both normals must be oriented *into* the object (pointing from the finger
    toward the object's interior). A pinch has the two sides pushing at each
    other, so their normals are anti-parallel and their dot product is near
    ``-1``; two fingers pressing the same face of an object too wide for the jaw
    to close on are parallel, dot near ``+1``. Bilateral contact alone does not
    distinguish those two, which is why contact count is not enough.

    Parameters
    ----------
    gripper_normal : numpy.ndarray or torch.Tensor
        ``(..., 3)`` inward contact normal for the fixed gripper finger,
        typically the force-weighted mean over that side's contacts. Need not be
        unit length. A zero vector (no contact) normalizes to zero and gives a
        dot of 0, so it is rejected for any positive ``threshold``; callers must
        still reject an empty side themselves, because ``threshold <= 0``
        accepts it.
    jaw_normal : numpy.ndarray or torch.Tensor
        ``(..., 3)`` inward contact normal for the moving jaw finger.
    threshold : float
        Required opposition in [-1, 1]
        (``RobotConfig.grasp_opposing_normal_threshold``). The test is
        ``dot <= -threshold``, so ``-1.0`` accepts any bilateral contact and
        ``1.0`` demands exact anti-parallelism.

    Returns
    -------
    numpy.bool_ or numpy.ndarray or torch.Tensor
        Boolean of the broadcast batch shape.
    """
    dot = (_normalize_last(gripper_normal) * _normalize_last(jaw_normal)).sum(-1)
    return dot <= -threshold
