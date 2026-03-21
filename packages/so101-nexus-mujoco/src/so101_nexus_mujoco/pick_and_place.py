"""MuJoCo pick-and-place environment.

Provides PickAndPlaceEnv where the robot must move a cube to a visible disc target.
"""

from __future__ import annotations

import tempfile
from typing import Literal

import mujoco
import numpy as np

from so101_nexus_core import get_so101_simulation_dir
from so101_nexus_core.config import (
    ControlMode,
    PickAndPlaceConfig,
)
from so101_nexus_core.constants import sample_color
from so101_nexus_core.rewards import reach_progress
from so101_nexus_core.observations import (
    EndEffectorPose,
    GraspState,
    ObjectOffset,
    ObjectPose,
    TargetOffset,
    TargetPosition,
)
from so101_nexus_mujoco.base_env import SO101NexusMuJoCoBaseEnv
from so101_nexus_mujoco.spawn_utils import random_yaw_quat

_SO101_DIR = get_so101_simulation_dir()
_SO101_XML = _SO101_DIR / "so101_new_calib.xml"


def _build_scene_xml(
    cube_half_size: float,
    cube_color: list[float],
    target_color: list[float],
    target_disc_radius: float,
    cube_mass: float,
    ground_color: list[float],
) -> str:
    r, g, b, a = cube_color
    tr, tg, tb, ta = target_color
    hs = cube_half_size
    robot_path = str(_SO101_XML)
    gr, gg, gb, ga = ground_color
    return f"""\
<mujoco model="pick_and_place_scene">
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

    <body name="cube" pos="0.15 0 {hs}">
      <freejoint name="cube_joint"/>
      <geom name="cube_geom" type="box" size="{hs} {hs} {hs}"
            rgba="{r} {g} {b} {a}" mass="{cube_mass}"
            contype="1" conaffinity="1" condim="4" friction="1 0.05 0.001"
            solref="0.01 1" solimp="0.95 0.99 0.001"/>
    </body>

    <body name="target" pos="0.15 0 0.001">
      <geom name="target_disc" type="cylinder" size="{target_disc_radius} 0.001"
            rgba="{tr} {tg} {tb} {ta}" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
"""


class PickAndPlaceEnv(SO101NexusMuJoCoBaseEnv):
    """Pick-and-place environment with a cube target and a visible goal marker."""

    config: PickAndPlaceConfig

    def __init__(
        self,
        config: PickAndPlaceConfig = PickAndPlaceConfig(),
        render_mode: str | None = None,
        camera_mode: Literal["state_only", "wrist"] = "state_only",
        control_mode: ControlMode = "pd_joint_pos",
        robot_init_qpos_noise: float = 0.02,
    ):
        if config.observations is None:
            config.observations = [
                EndEffectorPose(),
                GraspState(),
                TargetPosition(),
                ObjectPose(),
                ObjectOffset(),
                TargetOffset(),
            ]
        self._init_common(
            config=config,
            render_mode=render_mode,
            camera_mode=camera_mode,
            control_mode=control_mode,
            robot_init_qpos_noise=robot_init_qpos_noise,
        )

        cube_name = (
            config.cube_colors if isinstance(config.cube_colors, str) else config.cube_colors[0]
        )
        target_name = (
            config.target_colors
            if isinstance(config.target_colors, str)
            else config.target_colors[0]
        )
        self.cube_color_name = cube_name
        self.target_color_name = target_name
        self.cube_half_size = config.cube_half_size
        self.target_disc_radius = config.target_disc_radius
        self.task_description = (
            f"Pick up the small {cube_name} cube and place it on the {target_name} circle"
        )

        xml_string = _build_scene_xml(
            config.cube_half_size,
            sample_color(config.cube_colors),
            sample_color(config.target_colors),
            config.target_disc_radius,
            config.cube_mass,
            sample_color(config.ground_colors),
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", dir=_SO101_DIR, delete=True) as f:
            f.write(xml_string)
            f.flush()
            self.model = mujoco.MjModel.from_xml_path(f.name)
        self.data = mujoco.MjData(self.model)

        self._cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self._obj_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")

        cube_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
        self._cube_qpos_addr = self.model.jnt_qposadr[cube_joint_id]
        self._target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")

        self._finish_model_setup()

    def _get_cube_pose(self) -> np.ndarray:
        addr = self._cube_qpos_addr
        return self.data.qpos[addr : addr + 7].copy()

    def _get_target_pos(self) -> np.ndarray:
        return self.data.xpos[self._target_body_id].copy()

    def _get_obs(self) -> np.ndarray | dict:
        """Return proprioception plus cube and goal geometry, with an optional wrist image."""
        state = self._compute_obs_components()
        if self.camera_mode == "wrist":
            assert self._wrist_renderer is not None
            assert self._wrist_cam_id is not None
            self._wrist_renderer.update_scene(self.data, camera=self._wrist_cam_id)
            wrist_image = self._wrist_renderer.render()
            if self.config.obs_mode == "visual":
                self._privileged_state = state
                return {"state": self._get_current_qpos(), "wrist_camera": wrist_image}
            return {"state": state, "wrist_camera": wrist_image}
        return state

    def _get_component_data(self, component: object) -> np.ndarray:
        from so101_nexus_core.observations import (
            ObjectOffset as _ObjectOffset,
        )
        from so101_nexus_core.observations import (
            ObjectPose as _ObjectPose,
        )
        from so101_nexus_core.observations import (
            TargetOffset as _TargetOffset,
        )
        from so101_nexus_core.observations import (
            TargetPosition as _TargetPosition,
        )

        if isinstance(component, _ObjectPose):
            return self._get_cube_pose()
        if isinstance(component, _ObjectOffset):
            tcp_pos = self._get_tcp_pose()[:3]
            obj_pos = self._get_cube_pose()[:3]
            return obj_pos - tcp_pos
        if isinstance(component, _TargetPosition):
            return self._get_target_pos()
        if isinstance(component, _TargetOffset):
            obj_pos = self._get_cube_pose()[:3]
            target_pos = self._get_target_pos()
            return target_pos - obj_pos
        return super()._get_component_data(component)

    def _get_info(self) -> dict:
        tcp_pos = self._get_tcp_pose()[:3]
        obj_pose = self._get_cube_pose()
        obj_pos = obj_pose[:3]
        target_pos = self._get_target_pos()
        is_grasped = self._is_grasping()

        obj_to_target_xy = obj_pos[:2] - target_pos[:2]
        obj_to_target_dist = float(np.linalg.norm(obj_to_target_xy))
        is_obj_placed = (
            obj_to_target_dist <= self.config.goal_thresh
            and obj_pos[2] < self.cube_half_size + 0.01
        )
        is_robot_static = self._is_robot_static()
        lift_height = float(obj_pos[2] - self._initial_obj_z)
        success = is_obj_placed and is_robot_static

        info = {
            "obj_to_target_dist": obj_to_target_dist,
            "is_obj_placed": is_obj_placed,
            "is_grasped": is_grasped,
            "is_robot_static": is_robot_static,
            "lift_height": lift_height,
            "success": success,
            "tcp_to_obj_dist": float(np.linalg.norm(obj_pos - tcp_pos)),
        }
        if self._privileged_state is not None:
            info["privileged_state"] = self._privileged_state
        return info

    def _compute_reward(self, info: dict) -> float:
        scale = self.config.reward.tanh_shaping_scale
        rp = reach_progress(info["tcp_to_obj_dist"], scale=scale)
        is_grasped = info["is_grasped"] > 0.5
        placement_progress = (
            reach_progress(info["obj_to_target_dist"], scale=scale) if is_grasped else 0.0
        )
        return self.config.reward.compute(
            reach_progress=rp,
            is_grasped=is_grasped,
            task_progress=placement_progress,
            is_complete=info["success"],
            action_delta_norm=info.get("action_delta_norm", 0.0),
            energy_norm=info.get("energy_norm", 0.0),
        )

    def _task_reset(self) -> None:
        rng = self.np_random
        min_r = self.config.spawn_min_radius
        max_r = self.config.spawn_max_radius
        angle_half = float(np.radians(self.config.spawn_angle_half_range_deg))

        r_t = rng.uniform(min_r, max_r)
        theta_t = rng.uniform(-angle_half, angle_half)
        target_x = r_t * np.cos(theta_t)
        target_y = r_t * np.sin(theta_t)
        target_z = 0.001
        self.model.body_pos[self._target_body_id] = [target_x, target_y, target_z]

        for _ in range(100):
            r_c = rng.uniform(min_r, max_r)
            theta_c = rng.uniform(-angle_half, angle_half)
            cube_x = r_c * np.cos(theta_c)
            cube_y = r_c * np.sin(theta_c)
            dist = np.sqrt((cube_x - target_x) ** 2 + (cube_y - target_y) ** 2)
            if dist >= self.config.min_cube_target_separation:
                break

        cube_z = self.cube_half_size
        cube_quat = random_yaw_quat(rng)

        addr = self._cube_qpos_addr
        self.data.qpos[addr : addr + 3] = [cube_x, cube_y, cube_z]
        self.data.qpos[addr + 3 : addr + 7] = cube_quat
        self._initial_obj_z = cube_z
