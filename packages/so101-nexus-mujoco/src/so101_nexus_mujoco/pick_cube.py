from __future__ import annotations

import tempfile
from typing import Literal

import mujoco
import numpy as np

from so101_nexus_core import get_so101_simulation_dir
from so101_nexus_core.types import (
    CUBE_COLOR_MAP,
    DEFAULT_CAMERA_HEIGHT,
    DEFAULT_CAMERA_WIDTH,
    DEFAULT_CUBE_HALF_SIZE,
    DEFAULT_CUBE_MASS,
    DEFAULT_CUBE_SPAWN_HALF_SIZE,
    DEFAULT_GOAL_THRESH,
    DEFAULT_GROUND_COLOR,
    DEFAULT_LIFT_THRESHOLD,
    DEFAULT_MAX_GOAL_HEIGHT,
    ControlMode,
    CubeColorName,
    compute_normalized_reward,
)
from so101_nexus_mujoco.base_env import SO101NexusMuJoCoBaseEnv

_SO101_DIR = get_so101_simulation_dir()
_SO101_XML = _SO101_DIR / "so101_new_calib.xml"
_CUBE_SPAWN_CENTER = np.array([0.15, 0.0], dtype=np.float64)


def _build_scene_xml(cube_half_size: float, cube_color: list[float]) -> str:
    r, g, b, a = cube_color
    hs = cube_half_size
    robot_path = str(_SO101_XML)
    gr, gg, gb, ga = DEFAULT_GROUND_COLOR
    return f"""\
<mujoco model="pick_cube_scene">
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
            rgba="{r} {g} {b} {a}" mass="{DEFAULT_CUBE_MASS}"
            contype="1" conaffinity="1" friction="1 0.005 0.0001"
            solref="0.02 1" solimp="0.9 0.95 0.001"/>
    </body>

    <body name="goal" mocap="true" pos="0.15 0 {hs + 0.04}">
      <site name="goal_site" type="sphere" size="{DEFAULT_GOAL_THRESH}"
            rgba="0 1 0 0.5"/>
    </body>
  </worldbody>
</mujoco>
"""


class PickCubeEnv(SO101NexusMuJoCoBaseEnv):
    """MuJoCo pick-cube environment."""

    LIFT_THRESHOLD = DEFAULT_LIFT_THRESHOLD

    def __init__(
        self,
        cube_color: CubeColorName = "red",
        cube_half_size: float = DEFAULT_CUBE_HALF_SIZE,
        render_mode: str | None = None,
        camera_mode: Literal["state_only", "wrist"] = "state_only",
        camera_width: int = DEFAULT_CAMERA_WIDTH,
        camera_height: int = DEFAULT_CAMERA_HEIGHT,
        control_mode: ControlMode = "pd_joint_pos",
        robot_init_qpos_noise: float = 0.02,
    ):
        if cube_color not in CUBE_COLOR_MAP:
            raise ValueError(
                f"cube_color must be one of {list(CUBE_COLOR_MAP)}, got {cube_color!r}"
            )
        if not (0.01 <= cube_half_size <= 0.05):
            raise ValueError(f"cube_half_size must be in [0.01, 0.05], got {cube_half_size}")

        self._init_common(
            render_mode=render_mode,
            camera_mode=camera_mode,
            camera_width=camera_width,
            camera_height=camera_height,
            control_mode=control_mode,
            robot_init_qpos_noise=robot_init_qpos_noise,
        )

        self.cube_color_name = cube_color
        self.cube_half_size = cube_half_size
        self._goal_thresh = DEFAULT_GOAL_THRESH
        self.task_description = f"Pick up the small {cube_color} cube"

        xml_string = _build_scene_xml(cube_half_size, CUBE_COLOR_MAP[cube_color])
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", dir=_SO101_DIR, delete=True) as f:
            f.write(xml_string)
            f.flush()
            self.model = mujoco.MjModel.from_xml_path(f.name)
        self.data = mujoco.MjData(self.model)

        self._cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self._obj_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")

        cube_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
        self._cube_qpos_addr = self.model.jnt_qposadr[cube_joint_id]

        goal_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "goal")
        self._goal_mocap_id = self.model.body_mocapid[goal_body_id]

        self._finish_model_setup()

    def _get_cube_pose(self) -> np.ndarray:
        addr = self._cube_qpos_addr
        return self.data.qpos[addr : addr + 7].copy()

    def _get_goal_pos(self) -> np.ndarray:
        return self.data.mocap_pos[self._goal_mocap_id].copy()

    def _get_obs(self) -> np.ndarray | dict:
        tcp_pose = self._get_tcp_pose()
        is_grasped = np.array([self._is_grasping()])
        goal_pos = self._get_goal_pos()
        obj_pose = self._get_cube_pose()
        tcp_to_obj = obj_pose[:3] - tcp_pose[:3]
        obj_to_goal = goal_pos - obj_pose[:3]
        state = np.concatenate([tcp_pose, is_grasped, goal_pos, obj_pose, tcp_to_obj, obj_to_goal])

        if self.camera_mode == "wrist":
            self._wrist_renderer.update_scene(self.data, camera=self._wrist_cam_id)
            return {"state": state, "wrist_camera": self._wrist_renderer.render()}
        return state

    def _get_info(self) -> dict:
        tcp_pos = self._get_tcp_pose()[:3]
        obj_pose = self._get_cube_pose()
        obj_pos = obj_pose[:3]
        goal_pos = self._get_goal_pos()
        is_grasped = self._is_grasping()

        obj_to_goal_dist = float(np.linalg.norm(obj_pos - goal_pos))
        is_obj_placed = obj_to_goal_dist <= self._goal_thresh
        is_robot_static = self._is_robot_static()
        lift_height = float(obj_pos[2] - self._initial_obj_z)
        success = is_obj_placed and is_robot_static

        return {
            "obj_to_goal_dist": obj_to_goal_dist,
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
            (1.0 - float(np.tanh(5.0 * info["obj_to_goal_dist"]))) if is_grasped else 0.0
        )

        return compute_normalized_reward(
            reach_progress=reach_progress,
            is_grasped=is_grasped,
            task_progress=placement_progress,
            is_complete=info["success"],
        )

    def _task_reset(self) -> None:
        rng = self.np_random
        cx, cy = _CUBE_SPAWN_CENTER

        cube_x = cx + rng.uniform(-DEFAULT_CUBE_SPAWN_HALF_SIZE, DEFAULT_CUBE_SPAWN_HALF_SIZE)
        cube_y = cy + rng.uniform(-DEFAULT_CUBE_SPAWN_HALF_SIZE, DEFAULT_CUBE_SPAWN_HALF_SIZE)
        cube_z = self.cube_half_size

        angle = rng.uniform(0, 2 * np.pi)
        cube_quat = np.array([np.cos(angle / 2), 0, 0, np.sin(angle / 2)])

        addr = self._cube_qpos_addr
        self.data.qpos[addr : addr + 3] = [cube_x, cube_y, cube_z]
        self.data.qpos[addr + 3 : addr + 7] = cube_quat
        self._initial_obj_z = cube_z

        goal_x = cx + rng.uniform(-DEFAULT_CUBE_SPAWN_HALF_SIZE, DEFAULT_CUBE_SPAWN_HALF_SIZE)
        goal_y = cy + rng.uniform(-DEFAULT_CUBE_SPAWN_HALF_SIZE, DEFAULT_CUBE_SPAWN_HALF_SIZE)
        goal_z = self.cube_half_size + rng.uniform(0, DEFAULT_MAX_GOAL_HEIGHT)
        self.data.mocap_pos[self._goal_mocap_id] = [goal_x, goal_y, goal_z]


class PickCubeLiftEnv(PickCubeEnv):
    """Pick-cube variant where success requires lifting the cube above a threshold."""

    def _get_info(self) -> dict:
        info = super()._get_info()
        info["success"] = (info["lift_height"] > DEFAULT_LIFT_THRESHOLD) and (
            info["is_grasped"] > 0.5
        )
        return info

    def _compute_reward(self, info: dict) -> float:
        reach_progress = 1.0 - float(np.tanh(5.0 * info["tcp_to_obj_dist"]))
        is_grasped = info["is_grasped"] > 0.5
        lift_progress = float(np.tanh(5.0 * max(info["lift_height"], 0.0))) if is_grasped else 0.0

        return compute_normalized_reward(
            reach_progress=reach_progress,
            is_grasped=is_grasped,
            task_progress=lift_progress,
            is_complete=info["success"],
        )
