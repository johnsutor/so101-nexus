"""Batched grasp, lift, and return-to-rest task on MuJoCo Warp."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import mujoco
import numpy as np
import torch

from so101_nexus.config import ControlMode, PickReturnConfig
from so101_nexus.observations import RestingJointPositions, TargetOffset, TargetPosition
from so101_nexus.rewards import (
    pick_return_reach_potential,
    pick_return_success,
    pick_return_task_potential,
    potential_shaping,
)
from so101_nexus.warp.pick_env import WarpPickLiftVectorEnv

if TYPE_CHECKING:
    from so101_nexus.objects import SceneObject


class WarpPickReturnVectorEnv(WarpPickLiftVectorEnv):
    """Return each world's selected object to the configured resting arm posture."""

    config: PickReturnConfig
    default_config_cls: ClassVar[type[PickReturnConfig]] = PickReturnConfig

    def __init__(
        self,
        num_envs: int,
        config: PickReturnConfig | None = None,
        control_mode: ControlMode = "pd_joint_pos",
        device: str = "cuda",
        max_episode_steps: int = 1024,
        seed: int | None = None,
        nconmax: int | None = None,
        njmax: int | None = None,
        render_mode: str | None = None,
    ) -> None:
        if config is None:
            config = PickReturnConfig()
        if not isinstance(config, PickReturnConfig):
            raise TypeError("WarpPickReturnVectorEnv requires PickReturnConfig.")
        super().__init__(
            num_envs,
            config,
            control_mode,
            device,
            max_episode_steps,
            seed,
            nconmax,
            njmax,
            render_mode,
        )
        self._return_qpos_deg = torch.tensor(
            config.robot.rest_qpos_deg[:5], dtype=torch.float32, device=self.device
        )
        return_qpos_rad = np.radians(config.robot.rest_qpos_deg[:5])
        ctrl_bounds = self.mjm.actuator_ctrlrange[self._act_ids.cpu().numpy()[:5]]
        if np.any(return_qpos_rad < ctrl_bounds[:, 0]) or np.any(
            return_qpos_rad > ctrl_bounds[:, 1]
        ):
            raise ValueError("robot.rest_qpos_deg arm angles must be within control bounds")
        reference = mujoco.MjData(self.mjm)
        reference.qpos[self._qpos_adr[:5].cpu().numpy()] = return_qpos_rad
        mujoco.mj_kinematics(self.mjm, reference)
        self._return_tcp_pos = torch.tensor(
            reference.site_xpos[self._tcp_site_id], dtype=torch.float32, device=self.device
        )
        self._return_previous = tuple(torch.zeros(num_envs, device=self.device) for _ in range(3))

    def _describe_target(self, obj: SceneObject) -> str:
        return self.config.describe_target(obj)

    def _generic_task_description(self) -> str:
        return self.config.task_description

    def _supported_obs_components(self) -> set[type]:
        return super()._supported_obs_components() | {
            RestingJointPositions,
            TargetPosition,
            TargetOffset,
        }

    def _get_component_data(self, component: object) -> torch.Tensor:
        if isinstance(component, RestingJointPositions):
            return self._return_qpos_deg.expand(self.num_envs, -1)
        if isinstance(component, TargetPosition):
            return self._return_tcp_pos.expand(self.num_envs, -1)
        if isinstance(component, TargetOffset):
            return self._return_tcp_pos - self._tcp_pos()
        return super()._get_component_data(component)

    def _return_info(self) -> dict:
        grasped = self._is_grasping()
        lift_height = self._target_pos()[:, 2] - self._initial_obj_z
        rest_error_deg = (
            (torch.rad2deg(self._joint_qpos()[:, :5]) - self._return_qpos_deg).abs().amax(dim=1)
        )
        is_static = self._is_robot_static()
        return {
            "is_grasped": grasped,
            "is_robot_static": is_static,
            "lift_height": lift_height,
            "rest_error_deg": rest_error_deg,
            "arm_speed": self._joint_qvel()[:, :5].abs().amax(dim=1),
            "tcp_to_obj_dist": torch.linalg.norm(self._target_pos() - self._tcp_pos(), dim=1),
            "target_index": self._target_slot.clone(),
            "success": pick_return_success(
                lift_height,
                rest_error_deg,
                grasped,
                is_static,
                lift_threshold=self.config.lift_threshold,
                return_threshold_deg=self.config.return_threshold_deg,
            ),
        }

    def _return_potentials(self, info: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.config
        reach = pick_return_reach_potential(
            info["tcp_to_obj_dist"],
            info["is_grasped"],
            scale=cfg.reward.tanh_shaping_scale,
        )
        task = pick_return_task_potential(
            info["lift_height"],
            info["rest_error_deg"],
            info["arm_speed"],
            info["is_grasped"],
            lift_threshold=cfg.lift_threshold,
            return_threshold_deg=cfg.return_threshold_deg,
            scale=cfg.reward.tanh_shaping_scale,
            velocity_scale=cfg.reward.velocity_shaping_scale,
            max_goal_height=cfg.max_goal_height,
        )
        return reach, (info["is_grasped"] > 0.5).float(), task

    def _seed_return_potentials(self, mask: torch.Tensor) -> None:
        for previous, current in zip(
            self._return_previous, self._return_potentials(self._return_info()), strict=True
        ):
            previous[mask] = current[mask]

    def _refresh_reset_reference_state(self, mask: torch.Tensor) -> None:
        super()._refresh_reset_reference_state(mask)
        self._seed_return_potentials(mask)

    def reset(
        self,
        *,
        seed: int | list[int] | tuple[int, ...] | None = None,
        options: dict | None = None,
    ):
        """Seed reward potentials after settling, retaining Warp's spawn lift reference."""
        result = super().reset(seed=seed, options=options)
        self._seed_return_potentials(
            torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        )
        return result

    def _compute_reward_terminated(
        self,
        energy_norm: torch.Tensor,
        action_delta_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        info = self._return_info()
        current = self._return_potentials(info)
        deltas = [
            potential_shaping(now, prev)
            for now, prev in zip(current, self._return_previous, strict=True)
        ]
        self._return_previous = current
        components = self.config.reward.compute_components(
            *deltas,
            is_complete=info["success"],
            action_delta_norm=action_delta_norm,
            energy_norm=energy_norm,
        )
        info["reward_components"] = components
        return sum(components.values()).to(torch.float32), info["success"], info
