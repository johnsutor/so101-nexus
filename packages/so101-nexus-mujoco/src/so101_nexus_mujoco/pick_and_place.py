from __future__ import annotations

import tempfile
from typing import Literal

import mujoco
import numpy as np

from so101_nexus_core import get_so101_simulation_dir
from so101_nexus_core.config import (
    ControlMode,
    PickAndPlaceConfig,
    sample_color,
)
from so101_nexus_mujoco.base_env import SO101NexusMuJoCoBaseEnv

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
  <option timestep="0.002" gravity="0 0 -9.81"/>
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
            contype="1" conaffinity="1" friction="1 0.005 0.0001"
            solref="0.02 1" solimp="0.9 0.95 0.001"/>
    </body>

    <body name="target" pos="0.15 0 0.001">
      <geom name="target_disc" type="cylinder" size="{target_disc_radius} 0.001"
            rgba="{tr} {tg} {tb} {ta}" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
"""


class PickAndPlaceEnv(SO101NexusMuJoCoBaseEnv):
    """Pick-and-place environment with a visible coloured target disc on the ground."""

    config: PickAndPlaceConfig

    def __init__(
        self,
        config: PickAndPlaceConfig = PickAndPlaceConfig(),
        render_mode: str | None = None,
        camera_mode: Literal["state_only", "wrist"] = "state_only",
        control_mode: ControlMode = "pd_joint_pos",
        robot_init_qpos_noise: float = 0.02,
    ):
        if not (0.01 <= config.cube_half_size <= 0.05):
            raise ValueError(f"cube_half_size must be in [0.01, 0.05], got {config.cube_half_size}")

        self._init_common(
            config=config,
            render_mode=render_mode,
            camera_mode=camera_mode,
            control_mode=control_mode,
            robot_init_qpos_noise=robot_init_qpos_noise,
        )

        cube_name = config.cube_colors if isinstance(config.cube_colors, str) else config.cube_colors[0]
        target_name = config.target_colors if isinstance(config.target_colors, str) else config.target_colors[0]
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
        tcp_pose = self._get_tcp_pose()
        is_grasped = np.array([self._is_grasping()])
        target_pos = self._get_target_pos()
        obj_pose = self._get_cube_pose()
        tcp_to_obj = obj_pose[:3] - tcp_pose[:3]
        obj_to_target = target_pos - obj_pose[:3]
        state = np.concatenate(
            [tcp_pose, is_grasped, target_pos, obj_pose, tcp_to_obj, obj_to_target]
        )

        if self.camera_mode == "wrist":
            assert self._wrist_renderer is not None
            assert self._wrist_cam_id is not None
            self._wrist_renderer.update_scene(self.data, camera=self._wrist_cam_id)
            return {"state": state, "wrist_camera": self._wrist_renderer.render()}
        return state

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

        return {
            "obj_to_target_dist": obj_to_target_dist,
            "is_obj_placed": is_obj_placed,
            "is_grasped": is_grasped,
            "is_robot_static": is_robot_static,
            "lift_height": lift_height,
            "success": success,
            "tcp_to_obj_dist": float(np.linalg.norm(obj_pos - tcp_pos)),
        }

    def _compute_reward(self, info: dict) -> float:
        reach_progress = 1.0 - float(np.tanh(5.0 * info["tcp_to_obj_dist"]))
        is_grasped = info["is_grasped"] > 0.5
        placement_progress = (
            (1.0 - float(np.tanh(5.0 * info["obj_to_target_dist"]))) if is_grasped else 0.0
        )

        return self.config.reward.compute(
            reach_progress=reach_progress,
            is_grasped=is_grasped,
            task_progress=placement_progress,
            is_complete=info["success"],
            action_delta_norm=info.get("action_delta_norm", 0.0),
        )

    def _task_reset(self) -> None:
        rng = self.np_random
        cx, cy = self.config.spawn_center
        spawn_hs = self.config.spawn_half_size

        target_x = cx + rng.uniform(-spawn_hs, spawn_hs)
        target_y = cy + rng.uniform(-spawn_hs, spawn_hs)
        target_z = 0.001
        self.model.body_pos[self._target_body_id] = [target_x, target_y, target_z]

        for _ in range(100):
            cube_x = cx + rng.uniform(-spawn_hs, spawn_hs)
            cube_y = cy + rng.uniform(-spawn_hs, spawn_hs)
            dist = np.sqrt((cube_x - target_x) ** 2 + (cube_y - target_y) ** 2)
            if dist >= self.config.min_cube_target_separation:
                break

        cube_z = self.cube_half_size
        angle = rng.uniform(0, 2 * np.pi)
        cube_quat = np.array([np.cos(angle / 2), 0, 0, np.sin(angle / 2)])

        addr = self._cube_qpos_addr
        self.data.qpos[addr : addr + 3] = [cube_x, cube_y, cube_z]
        self.data.qpos[addr + 3 : addr + 7] = cube_quat
        self._initial_obj_z = cube_z
