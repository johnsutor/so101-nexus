"""Pick an object and return the SO-101 arm to its configured rest posture."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import mujoco
import numpy as np

from so101_nexus.config import ControlMode, PickReturnConfig
from so101_nexus.mujoco.pick_env import PickEnv
from so101_nexus.observations import RestingJointPositions, TargetOffset, TargetPosition
from so101_nexus.rewards import (
    pick_return_reach_potential,
    pick_return_success,
    pick_return_task_potential,
    potential_shaping,
)

if TYPE_CHECKING:
    from so101_nexus.objects import SceneObject


class PickReturnEnv(PickEnv):
    """Grasp, lift, and return to rest with low arm velocity."""

    config: PickReturnConfig
    default_config_cls: ClassVar[type[PickReturnConfig]] = PickReturnConfig
    _TASK_COMPONENTS = (*PickEnv._TASK_COMPONENTS, RestingJointPositions)

    def __init__(
        self,
        config: PickReturnConfig | None = None,
        render_mode: str | None = None,
        control_mode: ControlMode = "pd_joint_pos",
        robot_init_qpos_noise: float = 0.02,
    ) -> None:
        if config is None:
            config = PickReturnConfig()
        if not isinstance(config, PickReturnConfig):
            raise TypeError("PickReturnEnv requires PickReturnConfig.")
        super().__init__(config, render_mode, control_mode, robot_init_qpos_noise)
        self._return_qpos_deg = np.asarray(config.robot.rest_qpos_deg[:5], dtype=np.float64)
        self._return_qpos_rad = np.radians(self._return_qpos_deg)
        if np.any(self._return_qpos_rad < self._target_low[:5]) or np.any(
            self._return_qpos_rad > self._target_high[:5]
        ):
            raise ValueError("robot.rest_qpos_deg arm angles must be within control bounds")
        reference = mujoco.MjData(self.model)
        reference.qpos[self._arm_qpos_addrs] = self._return_qpos_rad
        mujoco.mj_kinematics(self.model, reference)
        self._return_tcp_pos = reference.site_xpos[self._tcp_site_id].copy()

    def _describe_target(self, target_obj: SceneObject) -> str:
        return self.config.describe_target(target_obj)

    def _get_component_data(self, component: object) -> np.ndarray:
        if isinstance(component, RestingJointPositions):
            return self._return_qpos_deg.copy()
        if isinstance(component, TargetPosition):
            return self._return_tcp_pos.copy()
        if isinstance(component, TargetOffset):
            return self._return_tcp_pos - self._get_tcp_pose()[:3]
        return super()._get_component_data(component)

    def _get_info(self) -> dict:
        info = super()._get_info()
        info["rest_error_deg"] = float(
            np.max(np.abs(np.degrees(self.data.qpos[self._arm_qpos_addrs]) - self._return_qpos_deg))
        )
        info["arm_speed"] = float(np.max(np.abs(self.data.qvel[self._arm_qvel_addrs])))
        info["success"] = pick_return_success(
            info["lift_height"],
            info["rest_error_deg"],
            info["is_grasped"],
            info["is_robot_static"],
            lift_threshold=self.config.lift_threshold,
            return_threshold_deg=self.config.return_threshold_deg,
        )
        return info

    def _return_potentials(self, info: dict) -> tuple[float, float, float]:
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
        return reach, float(info["is_grasped"] > 0.5), task

    def _refresh_reset_reference_state(self) -> None:
        super()._refresh_reset_reference_state()
        self._return_previous = self._return_potentials(self._get_info())

    def _compute_reward(self, info: dict) -> float:
        current = self._return_potentials(info)
        deltas = [
            potential_shaping(now, prev)
            for now, prev in zip(current, self._return_previous, strict=True)
        ]
        self._return_previous = current
        components = self.config.reward.compute_components(
            *deltas,
            is_complete=info["success"],
            action_delta_norm=info.get("action_delta_norm", 0.0),
            energy_norm=info.get("energy_norm", 0.0),
        )
        info["reward_components"] = components
        return sum(components.values())
