"""Primitive directional move environment for SO-101."""

from __future__ import annotations

import tempfile
from typing import Literal

import mujoco
import numpy as np

from so101_nexus_core import get_so101_simulation_dir
from so101_nexus_core.config import DIRECTION_VECTORS, ControlMode, MoveConfig
from so101_nexus_core.constants import sample_color
from so101_nexus_core.rewards import simple_reward
from so101_nexus_mujoco.base_env import SO101NexusMuJoCoBaseEnv

_SO101_DIR = get_so101_simulation_dir()
_SO101_XML = _SO101_DIR / "so101_new_calib.xml"


def _build_move_scene_xml(ground_rgba: list[float]) -> str:
    """Build MuJoCo XML string for the move scene (robot + floor + target site)."""
    robot_path = str(_SO101_XML)
    gr, gg, gb, ga = ground_rgba
    return f"""\
<mujoco model="move_scene">
  <option timestep="0.002" gravity="0 0 -9.81" cone="elliptic" noslip_iterations="3"/>
  <compiler angle="radian"/>

  <include file="{robot_path}"/>

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

    DO NOT call _is_grasping() in this env — there is no graspable object.
    """

    config: MoveConfig

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(
        self,
        config: MoveConfig | None = None,
        render_mode: str | None = None,
        camera_mode: Literal["state_only", "wrist"] = "state_only",
        control_mode: ControlMode = "pd_joint_pos",
        robot_init_qpos_noise: float = 0.02,
    ) -> None:
        if config is None:
            config = MoveConfig()
        self._init_common(
            config=config,
            render_mode=render_mode,
            camera_mode=camera_mode,
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
        return (
            f"Move the end-effector {self.config.direction} by {self.config.target_distance:.2f} m."
        )

    def _task_reset(self) -> None:
        # Forward kinematics is up-to-date because reset() calls mj_forward after _task_reset.
        # We use the current TCP position after joint reset (before mj_forward) so we
        # call mj_forward here to ensure site positions are updated.
        mujoco.mj_forward(self.model, self.data)
        initial_tcp_pos = self.data.site_xpos[self._tcp_site_id].copy()
        self._target_pos = initial_tcp_pos + self._dir_vec * self.config.target_distance
        # Keep target above floor
        self._target_pos[2] = max(self._target_pos[2], 0.02)
        self.model.site_pos[self._move_target_site_id] = self._target_pos

    def _get_component_data(self, component: object) -> np.ndarray:
        from so101_nexus_core.observations import TargetOffset

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
        return simple_reward(
            progress=progress,
            completion_bonus=self.config.reward.completion_bonus,
            success=info.get("success", False),
        )
