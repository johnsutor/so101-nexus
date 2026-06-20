"""Pure reward and progress functions shared across simulation backends.

``reach_progress`` and ``simple_reward`` accept Python floats, NumPy arrays, and
torch tensors, dispatching by duck typing so the scalar MuJoCo backend can call
them after scalar extraction and a tensor backend such as the planned MuJoCo
Warp backend can call the same formulas on batched tensors. ``torch`` is never
imported here; tensor support relies on operator overloading. ``orientation_progress``
remains scalar-only.
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


def orientation_progress(cos_similarity: float) -> float:
    """Map cosine similarity in [-1, 1] to a reward in [0, 1].

    Parameters
    ----------
    cos_similarity : float
        Cosine similarity between current and target direction.
        Values outside [-1, 1] are clamped.
    """
    clamped = max(-1.0, min(1.0, cos_similarity))
    return (clamped + 1.0) / 2.0


def simple_reward(*, progress, completion_bonus, success):
    """Reward for single-objective tasks (reach, move, look-at).

    Formula: ``(1 - completion_bonus) * progress + completion_bonus * success``.
    Accepts scalar or batched ``progress`` / ``success``; ``success`` may be a
    Python bool, NumPy bool array, or torch bool tensor (operator overloading
    promotes bool to float). The scalar path returns a plain ``float``.

    Parameters
    ----------
    progress : float or numpy.ndarray or torch.Tensor
        Task progress in [0, 1].
    completion_bonus : float
        Fraction of reward budget reserved for completion, in [0, 1].
    success : bool or numpy.ndarray or torch.Tensor
        Whether the task is complete.
    """
    return (1.0 - completion_bonus) * progress + completion_bonus * success
