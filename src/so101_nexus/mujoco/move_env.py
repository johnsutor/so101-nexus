"""Primitive directional move environment for SO-101."""

from __future__ import annotations

import tempfile
from typing import ClassVar

import mujoco
import numpy as np

from so101_nexus import get_so101_mujoco_model_dir, get_so101_mujoco_model_path
from so101_nexus.config import DIRECTION_VECTORS, ControlMode, MoveConfig
from so101_nexus.constants import sample_color
from so101_nexus.mujoco.base_env import SCENE_OPTION_XML, SO101NexusMuJoCoBaseEnv
from so101_nexus.rewards import simple_reward

_SO101_DIR = get_so101_mujoco_model_dir()
_SO101_XML = get_so101_mujoco_model_path()


def _build_move_scene_xml(ground_rgba: list[float]) -> str:
    """Build MuJoCo XML string for the move scene (robot + floor + target site)."""
    robot_path = str(_SO101_XML)
    gr, gg, gb, ga = ground_rgba
    return f"""\
<mujoco model="move_scene">
  <compiler angle="radian"/>

  <include file="{robot_path}"/>
  {SCENE_OPTION_XML}

  <visual>
    <headlight diffuse="0.0 0.0 0.0" ambient="0.3 0.3 0.3" specular="0 0 0"/>
  </visual>

  <worldbody>
    <light pos="1 1 3.5" dir="-0.27 -0.27 -0.92" directional="true" diffuse="0.5 0.5 0.5"/>
    <light pos="0 0 3.5" dir="0 0 -1" directional="true" diffuse="0.5 0.5 0.5"/>
    <geom name="floor" type="plane" size="0 0 0.01" rgba="{gr} {gg} {gb} {ga}"
          pos="0 0 0" contype="1" conaffinity="1"/>
    <site name="move_target" type="sphere" size="0.015" rgba="0 0.8 0.2 0.7"
          pos="0.15 0 0.1" group="1"/>
  </worldbody>
</mujoco>
"""


class MoveEnv(SO101NexusMuJoCoBaseEnv):
    """Move primitive: translate TCP a fixed distance in a specified direction.

    Default obs (6,): joint_positions.
    Info: tcp_to_target_dist, success.
    task_description: "Move the end-effector <direction> by <distance> m."

    DO NOT call _is_grasping() in this env - there is no graspable object.
    """

    config: MoveConfig
    default_config_cls: ClassVar[type[MoveConfig]] = MoveConfig

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(
        self,
        config: MoveConfig | None = None,
        render_mode: str | None = None,
        control_mode: ControlMode = "pd_joint_pos",
        robot_init_qpos_noise: float = 0.02,
    ) -> None:
        if config is None:
            config = MoveConfig()
        self._init_common(
            config=config,
            render_mode=render_mode,
            control_mode=control_mode,
            robot_init_qpos_noise=robot_init_qpos_noise,
        )

        ground_rgba = sample_color(config.ground_colors)
        xml_string = _build_move_scene_xml(ground_rgba)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", dir=_SO101_DIR, delete=True) as f:
            f.write(xml_string)
            f.flush()
            self.model = mujoco.MjModel.from_xml_path(f.name)
        self.data = mujoco.MjData(self.model)

        self._move_target_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "move_target"
        )
        self._target_pos: np.ndarray = np.zeros(3)
        self._dir_vec: np.ndarray = np.array(DIRECTION_VECTORS[config.direction], dtype=np.float64)

        self._finish_model_setup()

    @property
    def task_description(self) -> str:
        """Return a description of the current move task."""
        return self.config.task_description

    def _task_reset(self) -> None:
        # The move target is computed from the POST-settle TCP position so the
        # initial tcp_to_target_dist equals config.target_distance, following the
        # settle-then-place contract (settle before placing the target). The actual
        # placement happens in _refresh_reset_reference_state, the post-settle hook
        # base reset() calls after _settle_after_reset(). Nothing between _task_reset
        # and that hook reads the target site, so no provisional placement is needed.
        pass

    def _refresh_reset_reference_state(self) -> None:
        # Recompute the target from the settled TCP so the initial distance is
        # exactly config.target_distance (within tolerance).
        self._place_target_from_tcp()

    def _place_target_from_tcp(self) -> None:
        """Place the move target target_distance from the current TCP along the move direction."""
        tcp_pos = self.data.site_xpos[self._tcp_site_id].copy()
        self._target_pos = tcp_pos + self._dir_vec * self.config.target_distance
        # Keep target above floor. For downward move directions this clamp can
        # shorten the achievable distance, so the initial tcp_to_target_dist may be
        # less than config.target_distance. The clamp is part of the cross-backend
        # contract (consistent across backends).
        self._target_pos[2] = max(self._target_pos[2], 0.02)
        self.model.site_pos[self._move_target_site_id] = self._target_pos

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
