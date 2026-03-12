"""MuJoCo pick-YCB environment.

Provides PickYCBEnv and PickYCBLiftEnv for grasping real-world YCB objects
from the BenchBot object dataset.
"""
from __future__ import annotations

import tempfile
from typing import Literal

import mujoco
import numpy as np

from so101_nexus_core import (
    ensure_ycb_assets,
    get_so101_simulation_dir,
    get_ycb_collision_mesh,
    get_ycb_visual_mesh,
)
from so101_nexus_core.config import (
    YCB_OBJECTS,
    ControlMode,
    PickYCBConfig,
    YcbModelId,
    sample_color,
)
from so101_nexus_core.ycb_geometry import get_mujoco_ycb_rest_pose
from so101_nexus_mujoco.base_env import SO101NexusMuJoCoBaseEnv
from so101_nexus_mujoco.spawn_utils import random_yaw_quat

_SO101_DIR = get_so101_simulation_dir()
_SO101_XML = _SO101_DIR / "so101_new_calib.xml"


def _build_ycb_scene_xml(
    collision_mesh_path: str,
    visual_mesh_path: str,
    ground_color: list[float],
) -> str:
    robot_path = str(_SO101_XML)
    gr, gg, gb, ga = ground_color

    return f"""\
<mujoco model="pick_ycb_scene">
  <option timestep="0.002" gravity="0 0 -9.81" cone="elliptic" noslip_iterations="3"/>
  <compiler angle="radian"/>

  <include file="{robot_path}"/>

  <asset>
    <mesh name="ycb_collision" file="{collision_mesh_path}"/>
    <mesh name="ycb_visual" file="{visual_mesh_path}"/>
  </asset>

  <visual>
    <headlight diffuse="0.0 0.0 0.0" ambient="0.3 0.3 0.3" specular="0 0 0"/>
  </visual>

  <worldbody>
    <light pos="1 1 3.5" dir="-0.27 -0.27 -0.92" directional="true" diffuse="0.5 0.5 0.5"/>
    <light pos="0 0 3.5" dir="0 0 -1" directional="true" diffuse="0.5 0.5 0.5"/>
    <geom name="floor" type="plane" size="0 0 0.01" rgba="{gr} {gg} {gb} {ga}"
          pos="0 0 0" contype="1" conaffinity="1"/>

    <body name="ycb_object" pos="0.15 0 0.01">
      <freejoint name="obj_joint"/>
      <geom name="obj_collision" type="mesh" mesh="ycb_collision"
            mass="0.01" contype="1" conaffinity="1"
            condim="4" friction="1 0.05 0.001" solref="0.01 1" solimp="0.95 0.99 0.001"/>
      <geom name="obj_visual" type="mesh" mesh="ycb_visual"
            contype="0" conaffinity="0" mass="0"/>
    </body>
  </worldbody>
</mujoco>
"""


class PickYCBEnv(SO101NexusMuJoCoBaseEnv):
    """MuJoCo pick-YCB environment with goal-placement success."""

    config: PickYCBConfig

    def __init__(
        self,
        config: PickYCBConfig = PickYCBConfig(),
        model_id: YcbModelId = "058_golf_ball",
        render_mode: str | None = None,
        camera_mode: Literal["state_only", "wrist"] = "state_only",
        control_mode: ControlMode = "pd_joint_pos",
        robot_init_qpos_noise: float = 0.02,
    ):
        if model_id not in YCB_OBJECTS:
            raise ValueError(f"model_id must be one of {list(YCB_OBJECTS)}, got {model_id!r}")

        self._init_common(
            config=config,
            render_mode=render_mode,
            camera_mode=camera_mode,
            control_mode=control_mode,
            robot_init_qpos_noise=robot_init_qpos_noise,
        )

        self.model_id = model_id
        self.task_description = f"Pick up the {YCB_OBJECTS[model_id]}"

        ensure_ycb_assets(model_id)
        collision_path = str(get_ycb_collision_mesh(model_id))
        visual_path = str(get_ycb_visual_mesh(model_id))

        xml_string = _build_ycb_scene_xml(
            collision_path,
            visual_path,
            sample_color(config.ground_colors),
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", dir=_SO101_DIR, delete=True) as f:
            f.write(xml_string)
            f.flush()
            self.model = mujoco.MjModel.from_xml_path(f.name)
        self.data = mujoco.MjData(self.model)

        collision_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "obj_collision")
        mesh_id = self.model.geom_dataid[collision_geom_id]
        vert_start = self.model.mesh_vertadr[mesh_id]
        vert_count = self.model.mesh_vertnum[mesh_id]
        verts = self.model.mesh_vert[vert_start : vert_start + vert_count]
        self._obj_rest_quat, self._obj_spawn_z = get_mujoco_ycb_rest_pose(verts)

        self._obj_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ycb_object")
        self._obj_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "obj_collision")

        obj_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "obj_joint")
        self._obj_qpos_addr = self.model.jnt_qposadr[obj_joint_id]

        self._finish_model_setup()

    def _get_obj_pose(self) -> np.ndarray:
        addr = self._obj_qpos_addr
        return self.data.qpos[addr : addr + 7].copy()

    def _get_obs(self) -> np.ndarray | dict:
        tcp_pose = self._get_tcp_pose()
        is_grasped = np.array([self._is_grasping()])
        obj_pose = self._get_obj_pose()
        tcp_to_obj = obj_pose[:3] - tcp_pose[:3]
        state = np.concatenate([tcp_pose, is_grasped, obj_pose, tcp_to_obj])

        if self.camera_mode == "wrist":
            assert self._wrist_renderer is not None
            assert self._wrist_cam_id is not None
            self._wrist_renderer.update_scene(self.data, camera=self._wrist_cam_id)
            return {"state": state, "wrist_camera": self._wrist_renderer.render()}
        return state

    def _get_info(self) -> dict:
        tcp_pos = self._get_tcp_pose()[:3]
        obj_pose = self._get_obj_pose()
        obj_pos = obj_pose[:3]
        is_grasped = self._is_grasping()

        is_robot_static = self._is_robot_static()
        lift_height = float(obj_pos[2] - self._initial_obj_z)

        return {
            "is_grasped": is_grasped,
            "is_robot_static": is_robot_static,
            "lift_height": lift_height,
            "tcp_to_obj_dist": float(np.linalg.norm(obj_pos - tcp_pos)),
        }

    def _compute_reward(self, info: dict) -> float:
        return self._reach_only_reward(info)

    def _task_reset(self) -> None:
        rng = self.np_random
        min_r = self.config.spawn_min_radius
        max_r = self.config.spawn_max_radius
        angle_half = float(np.radians(self.config.spawn_angle_half_range_deg))

        r = rng.uniform(min_r, max_r)
        theta = rng.uniform(-angle_half, angle_half)
        obj_x = r * np.cos(theta)
        obj_y = r * np.sin(theta)
        obj_z = self._obj_spawn_z

        yaw_quat = random_yaw_quat(rng)
        obj_quat = np.zeros(4)
        mujoco.mju_mulQuat(obj_quat, yaw_quat, self._obj_rest_quat)

        addr = self._obj_qpos_addr
        self.data.qpos[addr : addr + 3] = [obj_x, obj_y, obj_z]
        self.data.qpos[addr + 3 : addr + 7] = obj_quat
        self._initial_obj_z = obj_z


class PickYCBLiftEnv(PickYCBEnv):
    """Pick-YCB variant where success requires lifting the object above threshold."""

    def _get_info(self) -> dict:
        info = super()._get_info()
        info["success"] = (info["lift_height"] > self.config.lift_threshold) and (
            info["is_grasped"] > 0.5
        )
        return info

    def _compute_reward(self, info: dict) -> float:
        return self._lift_reward(info)
