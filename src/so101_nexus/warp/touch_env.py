"""GPU-batched touch-an-object environment for SO-101 on MuJoCo Warp."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from so101_nexus.config import ControlMode, TouchConfig, describe_touch_target
from so101_nexus.rewards import reach_progress, simple_reward
from so101_nexus.warp.pick_env import WarpPickLiftVectorEnv

if TYPE_CHECKING:
    from so101_nexus.objects import SceneObject


class WarpTouchVectorEnv(WarpPickLiftVectorEnv):
    """Batched touch primitive: bring every world's TCP to its target object.

    Uses the same compiled object-slot pool as the pick backend, so cubes, YCB
    objects, and meshes (optionally among distractors) are valid targets. Success
    fires when the TCP reaches within the per-world target bounding radius plus
    ``touch_margin``.
    """

    config: TouchConfig

    def __init__(
        self,
        num_envs: int,
        config: TouchConfig | None = None,
        control_mode: ControlMode = "pd_joint_pos",
        device: str = "cuda",
        max_episode_steps: int = 512,
        seed: int | None = None,
        nconmax: int | None = None,
        njmax: int | None = None,
        render_mode: str | None = None,
    ) -> None:
        if config is None:
            config = TouchConfig()
        super().__init__(
            num_envs=num_envs,
            config=config,
            control_mode=control_mode,
            device=device,
            max_episode_steps=max_episode_steps,
            seed=seed,
            nconmax=nconmax,
            njmax=njmax,
            render_mode=render_mode,
        )

    def _describe_target(self, obj: SceneObject) -> str:
        return describe_touch_target(obj)

    def _generic_task_description(self) -> str:
        return "Touch the selected object."

    def _compute_reward_terminated(
        self, energy_norm: torch.Tensor, action_delta_norm: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        dist = torch.linalg.norm(self._target_pos() - self._tcp_pos(), dim=1)
        # Per-world threshold: reaching the object coincides with its bounding
        # radius (matches the MuJoCo touch backend) plus the configured margin.
        threshold = self._target_bounding_radius() + self.config.touch_margin
        # Tensor path of so101_nexus.rewards.reach_progress / simple_reward.
        progress = reach_progress(dist, scale=self.config.reward.tanh_shaping_scale)
        success = dist < threshold
        base = simple_reward(
            progress=progress,
            completion_bonus=self.config.reward.completion_bonus,
            success=success,
        )
        reward = self.config.reward.apply_penalties(
            base, action_delta_norm=action_delta_norm, energy_norm=energy_norm
        )
        info = {"tcp_to_obj_dist": dist, "success": success}
        return reward.to(torch.float32), success, info
