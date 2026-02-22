from __future__ import annotations

import tempfile
from typing import Literal

import gymnasium
import mujoco
import numpy as np
from gymnasium import spaces

from so101_nexus_core import get_so101_simulation_dir
from so101_nexus_core.types import CUBE_COLOR_MAP, CubeColorName

_SO101_DIR = get_so101_simulation_dir()
_SO101_XML = _SO101_DIR / "so101_new_calib.xml"

_REST_QPOS = np.array([0.0, -1.5708, 1.5708, 0.66, 0.0, -1.1], dtype=np.float64)

_CUBE_SPAWN_CENTER = np.array([0.15, 0.0], dtype=np.float64)
_CUBE_SPAWN_HALF_SIZE = 0.05
_MAX_GOAL_HEIGHT = 0.08
_GOAL_THRESH = 0.025
_LIFT_THRESHOLD = 0.05

_N_SUBSTEPS = 10

_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def _build_scene_xml(
    cube_half_size: float,
    cube_color: list[float],
) -> str:
    r, g, b, a = cube_color
    hs = cube_half_size
    robot_path = str(_SO101_XML)
    return f"""\
<mujoco model="pick_cube_scene">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <compiler angle="radian"/>

  <include file="{robot_path}"/>

  <worldbody>
    <light pos="0.2 0.2 0.8" dir="-0.2 -0.2 -0.8" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" type="plane" size="1 1 0.01" rgba="0.5 0.5 0.5 1"
          pos="0 0 0" contype="1" conaffinity="1"/>

    <body name="cube" pos="0.15 0 {hs}">
      <freejoint name="cube_joint"/>
      <geom name="cube_geom" type="box" size="{hs} {hs} {hs}"
            rgba="{r} {g} {b} {a}" mass="0.01"
            contype="1" conaffinity="1" friction="1 0.005 0.0001"
            solref="0.02 1" solimp="0.9 0.95 0.001"/>
    </body>

    <body name="goal" mocap="true" pos="0.15 0 {hs + 0.04}">
      <site name="goal_site" type="sphere" size="{_GOAL_THRESH}"
            rgba="0 1 0 0.5"/>
    </body>
  </worldbody>
</mujoco>
"""


class PickCubeEnv(gymnasium.Env):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 50}

    def __init__(
        self,
        cube_color: CubeColorName = "red",
        cube_half_size: float = 0.0125,
        render_mode: str | None = None,
        camera_mode: Literal["state_only", "wrist"] = "state_only",
        camera_width: int = 224,
        camera_height: int = 224,
    ):
        if cube_color not in CUBE_COLOR_MAP:
            raise ValueError(
                f"cube_color must be one of {list(CUBE_COLOR_MAP)}, got {cube_color!r}"
            )
        if not (0.01 <= cube_half_size <= 0.05):
            raise ValueError(f"cube_half_size must be in [0.01, 0.05], got {cube_half_size}")

        self.cube_color_name = cube_color
        self.cube_half_size = cube_half_size
        self.render_mode = render_mode
        self.camera_mode = camera_mode
        self.camera_width = camera_width
        self.camera_height = camera_height

        xml_string = _build_scene_xml(cube_half_size, CUBE_COLOR_MAP[cube_color])
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", dir=_SO101_DIR, delete=True) as f:
            f.write(xml_string)
            f.flush()
            self.model = mujoco.MjModel.from_xml_path(f.name)
        self.data = mujoco.MjData(self.model)

        self._joint_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in _JOINT_NAMES],
            dtype=np.int32,
        )
        self._actuator_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in _JOINT_NAMES],
            dtype=np.int32,
        )
        self._cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self._gripper_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
        self._jaw_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "moving_jaw_so101_v1"
        )
        self._tcp_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        self._cube_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")

        cube_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
        self._cube_qpos_addr = self.model.jnt_qposadr[cube_joint_id]

        goal_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "goal")
        self._goal_mocap_id = self.model.body_mocapid[goal_body_id]

        self._gripper_geom_ids = self._get_collision_geoms(self._gripper_body_id)
        self._jaw_geom_ids = self._get_collision_geoms(self._jaw_body_id)

        # Add finger pad geoms for better grasp detection
        static_pad_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "static_finger_pad")
        moving_pad_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "moving_finger_pad")
        if static_pad_id >= 0:
            self._gripper_geom_ids.add(static_pad_id)
        if moving_pad_id >= 0:
            self._jaw_geom_ids.add(moving_pad_id)

        self._arm_qvel_addrs = np.array(
            [self.model.jnt_dofadr[self._joint_ids[i]] for i in range(5)], dtype=np.int32
        )

        ctrl_range = self.model.actuator_ctrlrange[self._actuator_ids]
        self.action_space = spaces.Box(
            low=ctrl_range[:, 0].astype(np.float32),
            high=ctrl_range[:, 1].astype(np.float32),
            dtype=np.float32,
        )

        if self.camera_mode == "wrist":
            self._wrist_cam_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam"
            )
            self._wrist_renderer = mujoco.Renderer(
                self.model, height=self.camera_height, width=self.camera_width
            )
            self.observation_space = spaces.Dict(
                {
                    "state": spaces.Box(low=-np.inf, high=np.inf, shape=(24,), dtype=np.float64),
                    "wrist_camera": spaces.Box(
                        low=0,
                        high=255,
                        shape=(self.camera_height, self.camera_width, 3),
                        dtype=np.uint8,
                    ),
                }
            )
        else:
            self._wrist_cam_id = None
            self._wrist_renderer = None
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(24,), dtype=np.float64
            )

        self._renderer = None
        self._viewer = None

    def _get_collision_geoms(self, body_id: int) -> set[int]:
        geom_ids = set()
        for i in range(self.model.ngeom):
            if self.model.geom_bodyid[i] == body_id and self.model.geom_contype[i] != 0:
                geom_ids.add(i)
        return geom_ids

    def _get_tcp_pose(self) -> np.ndarray:
        pos = self.data.site_xpos[self._tcp_site_id].copy()
        mat = self.data.site_xmat[self._tcp_site_id].reshape(3, 3)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, mat.flatten())
        return np.concatenate([pos, quat])

    def _get_cube_pose(self) -> np.ndarray:
        addr = self._cube_qpos_addr
        return self.data.qpos[addr : addr + 7].copy()

    def _get_goal_pos(self) -> np.ndarray:
        return self.data.mocap_pos[self._goal_mocap_id].copy()

    def _is_grasping(self) -> float:
        gripper_contact = False
        jaw_contact = False

        force_buf = np.zeros(6)
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            g1, g2 = contact.geom1, contact.geom2

            cube_involved = g1 == self._cube_geom_id or g2 == self._cube_geom_id
            if not cube_involved:
                continue

            other = g2 if g1 == self._cube_geom_id else g1

            mujoco.mj_contactForce(self.model, self.data, i, force_buf)
            normal_force = abs(force_buf[0])

            if normal_force >= 0.5:
                if other in self._gripper_geom_ids:
                    gripper_contact = True
                if other in self._jaw_geom_ids:
                    jaw_contact = True

        return 1.0 if (gripper_contact and jaw_contact) else 0.0

    def _is_robot_static(self) -> bool:
        arm_vels = self.data.qvel[self._arm_qvel_addrs]
        return bool(np.all(np.abs(arm_vels) < 0.2))

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
            wrist_img = self._wrist_renderer.render()
            return {"state": state, "wrist_camera": wrist_img}

        return state

    def _get_info(self) -> dict:
        tcp_pos = self._get_tcp_pose()[:3]
        obj_pose = self._get_cube_pose()
        obj_pos = obj_pose[:3]
        goal_pos = self._get_goal_pos()
        is_grasped = self._is_grasping()

        obj_to_goal_dist = float(np.linalg.norm(obj_pos - goal_pos))
        is_obj_placed = obj_to_goal_dist <= _GOAL_THRESH
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
        tcp_to_obj_dist = info["tcp_to_obj_dist"]
        reaching = 1.0 - float(np.tanh(5.0 * tcp_to_obj_dist))

        reward = reaching

        is_grasped = info["is_grasped"]
        reward += is_grasped

        obj_to_goal_dist = info["obj_to_goal_dist"]
        placement = 1.0 - float(np.tanh(5.0 * obj_to_goal_dist))
        reward += placement * is_grasped

        reward += float(info["is_robot_static"]) * float(info["is_obj_placed"])

        if info["success"]:
            reward = 5.0

        return reward / 5.0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)

        mujoco.mj_resetData(self.model, self.data)

        for i, jid in enumerate(self._joint_ids):
            qpos_addr = self.model.jnt_qposadr[jid]
            self.data.qpos[qpos_addr] = _REST_QPOS[i]

        self.data.ctrl[self._actuator_ids] = _REST_QPOS

        rng = self.np_random
        cx, cy = _CUBE_SPAWN_CENTER
        cube_x = cx + rng.uniform(-_CUBE_SPAWN_HALF_SIZE, _CUBE_SPAWN_HALF_SIZE)
        cube_y = cy + rng.uniform(-_CUBE_SPAWN_HALF_SIZE, _CUBE_SPAWN_HALF_SIZE)
        cube_z = self.cube_half_size

        angle = rng.uniform(0, 2 * np.pi)
        cube_quat = np.array([np.cos(angle / 2), 0, 0, np.sin(angle / 2)])

        addr = self._cube_qpos_addr
        self.data.qpos[addr : addr + 3] = [cube_x, cube_y, cube_z]
        self.data.qpos[addr + 3 : addr + 7] = cube_quat

        self._initial_obj_z = cube_z

        goal_x = cx + rng.uniform(-_CUBE_SPAWN_HALF_SIZE, _CUBE_SPAWN_HALF_SIZE)
        goal_y = cy + rng.uniform(-_CUBE_SPAWN_HALF_SIZE, _CUBE_SPAWN_HALF_SIZE)
        goal_z = self.cube_half_size + rng.uniform(0, _MAX_GOAL_HEIGHT)

        self.data.mocap_pos[self._goal_mocap_id] = [goal_x, goal_y, goal_z]

        if self.camera_mode == "wrist":
            pitch = self.np_random.uniform(-0.6, 0.0)
            euler = np.array([pitch, 0.0, 2 * np.pi])
            quat = np.zeros(4)
            mujoco.mju_euler2Quat(quat, euler, "XYZ")
            mat = np.zeros(9)
            mujoco.mju_quat2Mat(mat, quat)
            self.model.cam_mat0[self._wrist_cam_id] = mat

            self.model.cam_pos0[self._wrist_cam_id] = [
                self.np_random.uniform(-0.005, 0.005),
                0.04 + self.np_random.uniform(-0.01, 0.01),
                -0.04 + self.np_random.uniform(-0.01, 0.01),
            ]

            self.model.cam_fovy[self._wrist_cam_id] = self.np_random.uniform(60, 90)

        mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.data.ctrl[self._actuator_ids] = action

        for _ in range(_N_SUBSTEPS):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        info = self._get_info()
        reward = self._compute_reward(info)
        terminated = bool(info["success"])

        return obs, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            self._renderer.update_scene(self.data)
            return self._renderer.render()
        elif self.render_mode == "human":
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
            return None
        return None

    def close(self):
        if self._wrist_renderer is not None:
            self._wrist_renderer.close()
            self._wrist_renderer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None


class PickCubeLiftEnv(PickCubeEnv):
    def _get_info(self) -> dict:
        info = super()._get_info()
        lift_height = info["lift_height"]
        is_grasped = info["is_grasped"]
        info["success"] = (lift_height > _LIFT_THRESHOLD) and (is_grasped > 0.5)
        return info

    def _compute_reward(self, info: dict) -> float:
        tcp_to_obj_dist = info["tcp_to_obj_dist"]
        reaching = 1.0 - float(np.tanh(5.0 * tcp_to_obj_dist))

        reward = reaching

        is_grasped = info["is_grasped"]
        reward += is_grasped

        lift_height = max(info["lift_height"], 0.0)
        lift_reward = float(np.tanh(5.0 * lift_height))
        reward += lift_reward * is_grasped

        if info["success"]:
            reward = 6.0

        return reward / 6.0
