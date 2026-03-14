"""Primitive reach-to-target environment for SO-101."""

from __future__ import annotations

import tempfile
from typing import Literal

import mujoco
import numpy as np

from so101_nexus_core import get_so101_simulation_dir
from so101_nexus_core.config import ControlMode, EnvironmentConfig, sample_color
from so101_nexus_mujoco.base_env import SO101NexusMuJoCoBaseEnv

_SO101_DIR = get_so101_simulation_dir()
_SO101_XML = _SO101_DIR / "so101_new_calib.xml"


def _build_reach_scene_xml(ground_rgba: list[float], target_radius: float) -> str:
    """Build the MuJoCo XML string for the reach scene (robot + floor + target site)."""
    robot_path = str(_SO101_XML)
    gr, gg, gb, ga = ground_rgba
    return f"""\
<mujoco model="reach_scene">
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
    <site name="reach_target" type="sphere" size="{target_radius}" rgba="1 0.5 0 0.7"
          pos="0.15 0 0.1" group="1"/>
  </worldbody>
</mujoco>
"""


class ReachConfig(EnvironmentConfig):
    """Config for the reach-to-target primitive task.

    Args:
        target_radius: Visual radius of the target site sphere (metres).
        target_workspace_half_extent: Half-width of the cubic workspace to
            sample target positions from (metres).
        success_threshold: TCP-to-target distance (m) that counts as success.
        **kwargs: Forwarded to EnvironmentConfig.
    """

    def __init__(
        self,
        target_radius: float = 0.02,
        target_workspace_half_extent: float = 0.15,
        success_threshold: float = 0.02,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.target_radius = target_radius
        self.target_workspace_half_extent = target_workspace_half_extent
        self.success_threshold = success_threshold


class ReachEnv(SO101NexusMuJoCoBaseEnv):
    """Reach primitive: move TCP to a randomly sampled 3-D target site.

    Obs (10,): tcp_pose(7) + tcp_to_target(3).
    Info: tcp_to_target_dist, success.

    DO NOT call _is_grasping() in this env — there is no graspable object.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}
    task_description = "Move the robot's end-effector to the target position."

    def __init__(
        self,
        config: ReachConfig | None = None,
        render_mode: str | None = None,
        camera_mode: Literal["state_only", "wrist"] = "state_only",
        control_mode: ControlMode = "pd_joint_pos",
        robot_init_qpos_noise: float = 0.02,
    ) -> None:
        if config is None:
            config = ReachConfig()
        self._init_common(
            config=config,
            render_mode=render_mode,
            camera_mode=camera_mode,
            control_mode=control_mode,
            robot_init_qpos_noise=robot_init_qpos_noise,
        )

        ground_rgba = sample_color(config.ground_colors)
        xml_string = _build_reach_scene_xml(ground_rgba, config.target_radius)
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

    def _state_obs_size(self) -> int:
        return 10

    def _task_reset(self) -> None:
        half = self.config.target_workspace_half_extent  # type: ignore[attr-defined]
        center = np.array([0.15, 0.0, 0.15])
        self._target_pos = center + self.np_random.uniform(-half, half, size=3)
        # Keep target above floor
        self._target_pos[2] = max(self._target_pos[2], 0.05)
        self.model.site_pos[self._target_site_id] = self._target_pos

    def _get_obs(self) -> np.ndarray:
        tcp_pose = self._get_tcp_pose()
        tcp_to_target = self._target_pos - tcp_pose[:3]
        return np.concatenate([tcp_pose, tcp_to_target])

    def _get_info(self) -> dict:
        tcp_pos = self._get_tcp_pose()[:3]
        dist = float(np.linalg.norm(self._target_pos - tcp_pos))
        return {
            "tcp_to_target_dist": dist,
            "success": dist < self.config.success_threshold,  # type: ignore[attr-defined]
        }

    def _compute_reward(self, info: dict) -> float:
        tcp_pos = self._get_tcp_pose()[:3]
        reach = self._reach_to_target_reward(tcp_pos, self._target_pos)
        completion_bonus = self.config.reward.completion_bonus
        bonus = completion_bonus if info.get("success", False) else 0.0
        return (1.0 - completion_bonus) * reach + bonus
