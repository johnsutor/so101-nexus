"""GPU-batched touch-an-object environment for SO-101 on MuJoCo Warp."""

from __future__ import annotations

import torch

from so101_nexus.config import ControlMode, TouchConfig, describe_touch_target
from so101_nexus.rewards import reach_progress, simple_reward
from so101_nexus.warp.pick_env import WarpPickLiftVectorEnv


class WarpTouchVectorEnv(WarpPickLiftVectorEnv):
    """Batched touch primitive: bring every world's TCP to its cube.

    Supports a single ``CubeObject`` target, matching the batched pick backend's
    limitation; the YCB/mesh objects and distractors available on
    ``MuJoCoTouch-v1`` are not supported here. Success fires when the TCP reaches
    within the cube's bounding radius plus ``touch_margin``.
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
        )
        # Bounding radius mirrors the MuJoCo backend's _cube_bounding_radius
        # (half_size * sqrt(2)); success coincides with reaching the cube.
        self._touch_threshold = self._cube.half_size * 2.0**0.5 + config.touch_margin
        self.task_description = describe_touch_target(self._cube)

    def _compute_reward_terminated(
        self, energy_norm: torch.Tensor, action_delta_norm: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        dist = torch.linalg.norm(self._cube_pos() - self._tcp_pos(), dim=1)
        # Tensor path of so101_nexus.rewards.reach_progress / simple_reward.
        progress = reach_progress(dist, scale=self.config.reward.tanh_shaping_scale)
        success = dist < self._touch_threshold
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
