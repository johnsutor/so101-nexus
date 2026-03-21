"""Pure reward and progress functions shared across simulation backends.

These operate on plain Python floats so both MuJoCo (numpy) and
ManiSkill (torch) backends can call them after scalar extraction,
or the formulas can be reimplemented in tensor form using the same logic.
"""

from __future__ import annotations

import math


def reach_progress(distance: float, *, scale: float) -> float:
    """Tanh-shaped progress in [0, 1]: 1 at distance 0, decaying toward 0.

    Negative distances are clamped to 0 (floating-point jitter tolerance).

    Parameters
    ----------
    distance : float
        Non-negative distance to target.
    scale : float
        Positive shaping steepness (``RewardConfig.tanh_shaping_scale``).
    """
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


def simple_reward(
    *,
    progress: float,
    completion_bonus: float,
    success: bool,
) -> float:
    """Reward for single-objective tasks (reach, move, look-at).

    Formula: ``(1 - completion_bonus) * progress + completion_bonus * success``

    Parameters
    ----------
    progress : float
        Task progress in [0, 1].
    completion_bonus : float
        Fraction of reward budget reserved for completion, in [0, 1].
        Caller (``RewardConfig``) is responsible for range validation.
    success : bool
        Whether the task is complete.
    """
    return (1.0 - completion_bonus) * progress + completion_bonus * float(success)
