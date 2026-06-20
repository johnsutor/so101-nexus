"""Primitive reach-to-target environment for SO-101."""

from __future__ import annotations

import tempfile
from typing import ClassVar

import mujoco
import numpy as np

from so101_nexus import get_so101_mujoco_model_dir, get_so101_mujoco_model_path
from so101_nexus.config import ControlMode, ReachConfig
from so101_nexus.constants import sample_color
from so101_nexus.mujoco.base_env import SO101NexusMuJoCoBaseEnv
from so101_nexus.rewards import simple_reward
from so101_nexus.scene import MUJOCO_SCENE_OPTION_XML, build_reach_scene_xml

_SO101_DIR = get_so101_mujoco_model_dir()
_SO101_XML = get_so101_mujoco_model_path()


class ReachEnv(SO101NexusMuJoCoBaseEnv):
    """Reach primitive: move TCP to a randomly sampled 3-D target site.

    Default obs (6,): joint_positions.
    Info: tcp_to_target_dist, success.

    DO NOT call _is_grasping() in this env - there is no graspable object.
    """

    config: ReachConfig
    default_config_cls: ClassVar[type[ReachConfig]] = ReachConfig

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}
    task_description = "Move the robot's end-effector to the target position."

    def __init__(
        self,
        config: ReachConfig | None = None,
        render_mode: str | None = None,
        control_mode: ControlMode = "pd_joint_pos",
        robot_init_qpos_noise: float = 0.02,
    ) -> None:
        if config is None:
            config = ReachConfig()
        self._init_common(
            config=config,
            render_mode=render_mode,
            control_mode=control_mode,
            robot_init_qpos_noise=robot_init_qpos_noise,
        )

        ground_rgba = sample_color(config.ground_colors)
        xml_string = build_reach_scene_xml(
            ground_rgba,
            config.target_radius,
            option_xml=MUJOCO_SCENE_OPTION_XML,
            robot_xml_path=str(_SO101_XML),
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", dir=_SO101_DIR, delete=True) as f:
            f.write(xml_string)
            f.flush()
            self.model = mujoco.MjModel.from_xml_path(f.name)
        self.data = mujoco.MjData(self.model)

        self._target_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "reach_target"
        )
        self._target_pos: np.ndarray = np.zeros(3)

        self._finish_model_setup()

    def _task_reset(self) -> None:
        # Sample a 3-D cubic workspace per the cross-backend reach contract:
        # center = [0.15, 0.0, 0.15], per-axis offset in [-half, +half], z clamped
        # to >= 0.05. The MuJoCo target is a visual marker, so it may sit above the
        # floor (z is not forced to the floor radius).
        rng = self.np_random
        half = self.config.target_workspace_half_extent
        center = np.array([0.15, 0.0, 0.15])
        offset = rng.uniform(-half, half, size=3)
        pos = center + offset
        pos[2] = max(pos[2], 0.05)

        self._target_pos = pos
        self.model.site_pos[self._target_site_id] = self._target_pos

    def _get_component_data(self, component: object) -> np.ndarray:
        from so101_nexus.observations import TargetOffset

        if isinstance(component, TargetOffset):
            return self._target_pos - self._get_tcp_pose()[:3]
        return super()._get_component_data(component)

    def _get_info(self) -> dict:
        tcp_pos = self._get_tcp_pose()[:3]
        dist = float(np.linalg.norm(self._target_pos - tcp_pos))
        info = {
            "tcp_to_target_dist": dist,
            "success": dist < self.config.success_threshold,
        }
        if self._privileged_state is not None:
            info["privileged_state"] = self._privileged_state
        return info

    def _compute_reward(self, info: dict) -> float:
        tcp_pos = self._get_tcp_pose()[:3]
        progress = self._reach_to_target_reward(tcp_pos, self._target_pos)
        base = simple_reward(
            progress=progress,
            completion_bonus=self.config.reward.completion_bonus,
            success=info.get("success", False),
        )
        return self.config.reward.apply_penalties(
            base,
            action_delta_norm=info.get("action_delta_norm", 0.0),
            energy_norm=info.get("energy_norm", 0.0),
        )
