"""Normalized reward budget."""

from __future__ import annotations

import math


class RewardConfig:
    """Normalized reward budget.

    The four component weights must sum to 1.0. Penalty terms are applied
    additively and subtracted from the base reward.

    Args:
        reaching: Weight for TCP-to-object distance shaping.
        grasping: Weight for grasp binary signal.
        task_objective: Weight for task-specific progress.
        completion_bonus: Weight for episode completion signal.
        action_delta_penalty: Penalty coefficient on L2 norm of consecutive action deltas.
        energy_penalty: Penalty coefficient on L2 norm of the action vector (energy cost).
        tanh_shaping_scale: Scale factor for tanh distance shaping.
    """

    def __init__(
        self,
        reaching: float = 0.25,
        grasping: float = 0.25,
        task_objective: float = 0.40,
        completion_bonus: float = 0.10,
        action_delta_penalty: float = 0.0,
        energy_penalty: float = 0.0,
        tanh_shaping_scale: float = 5.0,
    ) -> None:
        total = reaching + grasping + task_objective + completion_bonus
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(f"Reward weights must sum to 1.0, got {total:.6f}")
        self.reaching = reaching
        self.grasping = grasping
        self.task_objective = task_objective
        self.completion_bonus = completion_bonus
        self.action_delta_penalty = action_delta_penalty
        self.energy_penalty = energy_penalty
        self.tanh_shaping_scale = tanh_shaping_scale

    def compute(
        self,
        reach_progress: float,
        is_grasped: bool,
        task_progress: float,
        is_complete: bool,
        action_delta_norm: float = 0.0,
        energy_norm: float = 0.0,
    ) -> float:
        """Compute a normalized reward using this config's weights."""
        base = (
            self.reaching * reach_progress
            + self.grasping * float(is_grasped)
            + self.task_objective * task_progress
            + self.completion_bonus * float(is_complete)
        )
        penalty = self.action_delta_penalty * action_delta_norm + self.energy_penalty * energy_norm
        return base - penalty

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"RewardConfig(reaching={self.reaching}, grasping={self.grasping}, "
            f"task_objective={self.task_objective}, completion_bonus={self.completion_bonus}, "
            f"action_delta_penalty={self.action_delta_penalty}, "
            f"energy_penalty={self.energy_penalty})"
        )
