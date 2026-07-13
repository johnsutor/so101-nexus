"""Primitive touch-an-object environment for SO-101."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np

from so101_nexus.config import ControlMode, TouchConfig, describe_touch_target
from so101_nexus.mujoco.pick_env import PickEnv
from so101_nexus.rewards import reach_progress

if TYPE_CHECKING:
    from so101_nexus.objects import SceneObject


class TouchEnv(PickEnv):
    """Touch primitive: bring the gripper to an object grounded on the table.

    Reuses the unified pick scene, so the target can be any cube, YCB object
    (for example a spoon), or mesh, optionally among distractors. Unlike the old
    reach marker that floated in free space, the target rests on the table, so
    its depth is unambiguous in the camera. Success fires when the TCP reaches
    within the target's bounding radius plus ``touch_margin``.

    Default obs (24,): joint_positions + end_effector_pose + grasp_state +
    object_pose + object_offset.
    Info: tcp_to_obj_dist, success.
    task_description is auto-generated: "Touch the <repr(object)>."
    """

    config: TouchConfig
    default_config_cls: ClassVar[type[TouchConfig]] = TouchConfig

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(
        self,
        config: TouchConfig | None = None,
        render_mode: str | None = None,
        control_mode: ControlMode = "pd_joint_pos",
        robot_init_qpos_noise: float = 0.02,
    ) -> None:
        if config is None:
            config = TouchConfig()
        super().__init__(
            config=config,
            render_mode=render_mode,
            control_mode=control_mode,
            robot_init_qpos_noise=robot_init_qpos_noise,
        )

    def _describe_target(self, target_obj: SceneObject) -> str:
        return describe_touch_target(target_obj)

    def _get_info(self) -> dict:
        tcp_pos = self._get_tcp_pose()[:3]
        obj_pos = self._get_target_pose()[:3]
        dist = float(np.linalg.norm(obj_pos - tcp_pos))
        threshold = self._target_bounding_radius() + self.config.touch_margin
        info = {"tcp_to_obj_dist": dist, "success": dist < threshold}
        if self._privileged_state is not None:
            info["privileged_state"] = self._privileged_state
        return info

    def _compute_reward(self, info: dict) -> float:
        progress = reach_progress(
            info["tcp_to_obj_dist"], scale=self.config.reward.tanh_shaping_scale
        )
        components = self.config.reward.compute_simple_components(
            progress,
            info.get("success", False),
            progress_key="reaching",
            action_delta_norm=info.get("action_delta_norm", 0.0),
            energy_norm=info.get("energy_norm", 0.0),
        )
        info["reward_components"] = components
        return sum(components.values())
