"""Pure reward and progress functions shared across simulation backends.

``reach_progress``, ``orientation_progress``, ``lift_progress``, and
``simple_reward`` accept Python floats, NumPy arrays, and torch tensors,
dispatching by duck typing so the scalar MuJoCo backend can call them after
scalar extraction and the batched MuJoCo Warp backend can call the same formulas
on tensors. ``torch`` is never imported here; tensor support relies on operator
overloading.
"""

from __future__ import annotations

import math

import numpy as np


def reach_progress(distance, *, scale):
    """Tanh-shaped progress in [0, 1]: 1 at distance 0, decaying toward 0.

    Accepts a Python float, NumPy array, or torch tensor. Negative distances are
    clamped to 0. The scalar path returns a plain ``float`` and is bit-identical
    to the original implementation.

    Parameters
    ----------
    distance : float or numpy.ndarray or torch.Tensor
        Non-negative distance(s) to target.
    scale : float
        Positive shaping steepness (``RewardConfig.tanh_shaping_scale``).
    """
    if hasattr(distance, "tanh"):  # torch.Tensor
        return 1.0 - (scale * distance.clamp(min=0.0)).tanh()
    if isinstance(distance, np.ndarray):
        return 1.0 - np.tanh(scale * np.clip(distance, 0.0, None))
    return 1.0 - math.tanh(scale * max(0.0, distance))


def orientation_progress(cos_similarity):
    """Map cosine similarity in [-1, 1] to a reward in [0, 1].

    Accepts a Python float, NumPy array, or torch tensor (dispatch by duck
    typing, like ``reach_progress``). Values outside [-1, 1] are clamped. The
    scalar path returns a plain ``float``.

    Parameters
    ----------
    cos_similarity : float or numpy.ndarray or torch.Tensor
        Cosine similarity between current and target direction.
    """
    if hasattr(cos_similarity, "clamp"):  # torch.Tensor
        return (cos_similarity.clamp(-1.0, 1.0) + 1.0) / 2.0
    if isinstance(cos_similarity, np.ndarray):
        return (np.clip(cos_similarity, -1.0, 1.0) + 1.0) / 2.0
    return (max(-1.0, min(1.0, cos_similarity)) + 1.0) / 2.0


def lift_progress(height, *, scale, grasped):
    """Tanh-shaped lift progress in [0, 1], zero unless grasped.

    Formula: ``tanh(scale * max(height, 0)) * grasped``. Accepts a Python float,
    NumPy array, or torch tensor for ``height``; ``grasped`` is a matching
    bool/float scalar or batch (operator overloading promotes bool to float).
    The scalar path returns a plain ``float`` and is bit-identical to the inline
    lift shaping it replaces.

    Parameters
    ----------
    height : float or numpy.ndarray or torch.Tensor
        Lift height above the per-episode baseline (metres).
    scale : float
        Positive shaping steepness (``RewardConfig.tanh_shaping_scale``).
    grasped : bool or numpy.ndarray or torch.Tensor
        Whether the object is currently grasped; ungrasped contributes 0.
    """
    if hasattr(height, "tanh"):  # torch.Tensor
        return (scale * height.clamp(min=0.0)).tanh() * grasped
    if isinstance(height, np.ndarray):
        return np.tanh(scale * np.clip(height, 0.0, None)) * grasped
    return math.tanh(scale * max(0.0, height)) * float(grasped)


def simple_reward(*, progress, completion_bonus, success):
    """Reward for single-objective tasks (reach, move, look-at).

    Shaping fills ``[0, 1 - completion_bonus]`` via ``progress``; success lifts
    the reward to the full budget (1.0), so a successful terminal step is always
    the global maximum with ``completion_bonus`` as the guaranteed margin. This
    matches the terminal clamp in ``RewardConfig.compute``. Accepts scalar or
    batched ``progress`` / ``success``; ``success`` may be a Python bool, NumPy
    bool array, or torch bool tensor (operator overloading promotes bool to
    float). The scalar path returns a plain ``float``.

    Parameters
    ----------
    progress : float or numpy.ndarray or torch.Tensor
        Task progress in [0, 1].
    completion_bonus : float
        Fraction of reward budget reserved for completion, in [0, 1].
    success : bool or numpy.ndarray or torch.Tensor
        Whether the task is complete.
    """
    shaped = (1.0 - completion_bonus) * progress
    return shaped + (1.0 - shaped) * success
