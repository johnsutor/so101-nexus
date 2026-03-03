from __future__ import annotations

import tempfile
from typing import Literal

import gymnasium
import mujoco
import numpy as np
from gymnasium import spaces

from so101_nexus_core import (
    ensure_ycb_assets,
    get_so101_simulation_dir,
    get_ycb_collision_mesh,
    get_ycb_visual_mesh,
)
from so101_nexus_core.types import (
    DEFAULT_CUBE_SPAWN_HALF_SIZE,
    DEFAULT_GOAL_THRESH,
    DEFAULT_GROUND_COLOR,
    DEFAULT_LIFT_THRESHOLD,
    DEFAULT_MAX_GOAL_HEIGHT,
    SO101_JOINT_NAMES,
    SO101_REST_QPOS,
    YCB_OBJECTS,
    ControlMode,
    compute_normalized_reward,
)

_SO101_DIR = get_so101_simulation_dir()
_SO101_XML = _SO101_DIR / "so101_new_calib.xml"

_REST_QPOS = np.array(SO101_REST_QPOS, dtype=np.float64)

_CUBE_SPAWN_CENTER = np.array([0.15, 0.0], dtype=np.float64)

_N_SUBSTEPS = 10


def _build_ycb_scene_xml(
    collision_mesh_path: str,
    visual_mesh_path: str,
    goal_mode: bool = True,
) -> str:
    """Build MuJoCo XML for a pick-YCB scene.

    Parameters
    ----------
    collision_mesh_path : str
        Absolute path to the collision .obj mesh.
    visual_mesh_path : str
        Absolute path to the visual .obj mesh.
    goal_mode : bool
        If True, include a mocap goal sphere.
    """
    robot_path = str(_SO101_XML)
    gr, gg, gb, ga = DEFAULT_GROUND_COLOR

    goal_xml = ""
    if goal_mode:
        goal_xml = f"""
    <body name="goal" mocap="true" pos="0.15 0 0.04">
      <site name="goal_site" type="sphere" size="{DEFAULT_GOAL_THRESH}"
            rgba="0 1 0 0.5"/>
    </body>"""

    return f"""\
<mujoco model="pick_ycb_scene">
  <option timestep="0.002" gravity="0 0 -9.81"/>
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
            friction="1 0.005 0.0001" solref="0.02 1" solimp="0.9 0.95 0.001"/>
      <geom name="obj_visual" type="mesh" mesh="ycb_visual"
            contype="0" conaffinity="0" mass="0"/>
    </body>
{goal_xml}
  </worldbody>
</mujoco>
"""


class PickYCBEnv(gymnasium.Env):
    """MuJoCo pick-YCB environment with goal-placement success.

    The agent must reach and grasp a YCB object, then move it to a goal
    position indicated by a floating transparent sphere.
    """

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 50}

    _VALID_CONTROL_MODES: set[str] = {
        "pd_joint_pos",
        "pd_joint_delta_pos",
        "pd_joint_target_delta_pos",
    }

    def __init__(
        self,
        model_id: str = "058_golf_ball",
        render_mode: str | None = None,
        camera_mode: Literal["state_only", "wrist"] = "state_only",
        camera_width: int = 224,
        camera_height: int = 224,
        control_mode: ControlMode = "pd_joint_pos",
    ):
        if model_id not in YCB_OBJECTS:
            raise ValueError(f"model_id must be one of {list(YCB_OBJECTS)}, got {model_id!r}")
        if control_mode not in self._VALID_CONTROL_MODES:
            valid = sorted(self._VALID_CONTROL_MODES)
            raise ValueError(f"control_mode must be one of {valid}, got {control_mode!r}")

        self.model_id = model_id
        self.control_mode = control_mode
        self.render_mode = render_mode
        self.camera_mode = camera_mode
        self.camera_width = camera_width
        self.camera_height = camera_height

        ensure_ycb_assets(model_id)
        collision_path = str(get_ycb_collision_mesh(model_id))
        visual_path = str(get_ycb_visual_mesh(model_id))

        xml_string = _build_ycb_scene_xml(collision_path, visual_path, goal_mode=True)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", dir=_SO101_DIR, delete=True) as f:
            f.write(xml_string)
            f.flush()
            self.model = mujoco.MjModel.from_xml_path(f.name)
        self.data = mujoco.MjData(self.model)

        # Compute a stable resting orientation and spawn Z.
        # Many YCB meshes have their longest axis along Z (e.g. fork handle).
        # We rotate so the thinnest axis points up, then compute spawn Z
        # from the rotated AABB so the object rests flat on the ground.
        collision_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "obj_collision")
        mesh_id = self.model.geom_dataid[collision_geom_id]
        vert_start = self.model.mesh_vertadr[mesh_id]
        vert_count = self.model.mesh_vertnum[mesh_id]
        verts = self.model.mesh_vert[vert_start : vert_start + vert_count]

        extents = np.ptp(verts, axis=0)  # [x_ext, y_ext, z_ext]
        thin_axis = int(np.argmin(extents))

        if thin_axis == 2:
            # Z is already thinnest — spawn upright, no extra rotation
            self._obj_rest_quat = np.array([1.0, 0.0, 0.0, 0.0])
            self._obj_spawn_z = float(-np.min(verts[:, 2])) + 0.002
        elif thin_axis == 0:
            # X is thinnest — rotate 90° about Y so X→Z
            self._obj_rest_quat = np.array([0.7071068, 0.0, 0.7071068, 0.0])
            rotated = verts[:, [2, 1, 0]].copy()
            rotated[:, 0] *= -1  # 90° about Y: x'=-z, z'=x
            self._obj_spawn_z = float(-np.min(rotated[:, 2])) + 0.002
        else:
            # Y is thinnest — rotate 90° about X so Y→Z
            self._obj_rest_quat = np.array([0.7071068, 0.7071068, 0.0, 0.0])
            rotated = verts[:, [0, 2, 1]].copy()
            rotated[:, 1] *= -1  # 90° about X: y'=-z, z'=y
            self._obj_spawn_z = float(-np.min(rotated[:, 2])) + 0.002

        self._joint_ids = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
                for n in SO101_JOINT_NAMES
            ],
            dtype=np.int32,
        )
        self._actuator_ids = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                for n in SO101_JOINT_NAMES
            ],
            dtype=np.int32,
        )
        self._obj_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ycb_object")
        self._gripper_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
        self._jaw_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "moving_jaw_so101_v1"
        )
        self._tcp_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        self._obj_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "obj_collision")

        obj_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "obj_joint")
        self._obj_qpos_addr = self.model.jnt_qposadr[obj_joint_id]

        goal_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "goal")
        self._goal_mocap_id = self.model.body_mocapid[goal_body_id]

        self._gripper_geom_ids = self._get_collision_geoms(self._gripper_body_id)
        self._jaw_geom_ids = self._get_collision_geoms(self._jaw_body_id)

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
        self._ctrl_low = ctrl_range[:, 0].copy()
        self._ctrl_high = ctrl_range[:, 1].copy()

        if self.control_mode == "pd_joint_pos":
            self.action_space = spaces.Box(
                low=self._ctrl_low.astype(np.float32),
                high=self._ctrl_high.astype(np.float32),
                dtype=np.float32,
            )
        else:
            delta_low = np.array([-0.05, -0.05, -0.05, -0.05, -0.05, -0.2], dtype=np.float32)
            delta_high = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.2], dtype=np.float32)
            self.action_space = spaces.Box(low=delta_low, high=delta_high, dtype=np.float32)

        self._prev_target: np.ndarray | None = None

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

    def _get_obj_pose(self) -> np.ndarray:
        addr = self._obj_qpos_addr
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

            obj_involved = g1 == self._obj_geom_id or g2 == self._obj_geom_id
            if not obj_involved:
                continue

            other = g2 if g1 == self._obj_geom_id else g1

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
        obj_pose = self._get_obj_pose()
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
        obj_pose = self._get_obj_pose()
        obj_pos = obj_pose[:3]
        goal_pos = self._get_goal_pos()
        is_grasped = self._is_grasping()

        obj_to_goal_dist = float(np.linalg.norm(obj_pos - goal_pos))
        is_obj_placed = obj_to_goal_dist <= DEFAULT_GOAL_THRESH
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
        reach_progress = 1.0 - float(np.tanh(5.0 * tcp_to_obj_dist))

        is_grasped = info["is_grasped"] > 0.5

        obj_to_goal_dist = info["obj_to_goal_dist"]
        placement_progress = (1.0 - float(np.tanh(5.0 * obj_to_goal_dist))) if is_grasped else 0.0

        is_complete = info["success"] and info["is_robot_static"]

        return compute_normalized_reward(
            reach_progress=reach_progress,
            is_grasped=is_grasped,
            task_progress=placement_progress,
            is_complete=is_complete,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)

        mujoco.mj_resetData(self.model, self.data)

        for i, jid in enumerate(self._joint_ids):
            qpos_addr = self.model.jnt_qposadr[jid]
            self.data.qpos[qpos_addr] = _REST_QPOS[i]

        self.data.ctrl[self._actuator_ids] = _REST_QPOS

        rng = self.np_random
        cx, cy = _CUBE_SPAWN_CENTER
        obj_x = cx + rng.uniform(-DEFAULT_CUBE_SPAWN_HALF_SIZE, DEFAULT_CUBE_SPAWN_HALF_SIZE)
        obj_y = cy + rng.uniform(-DEFAULT_CUBE_SPAWN_HALF_SIZE, DEFAULT_CUBE_SPAWN_HALF_SIZE)
        obj_z = self._obj_spawn_z

        # Compose random yaw with the rest orientation that lays the object flat
        angle = rng.uniform(0, 2 * np.pi)
        yaw_quat = np.array([np.cos(angle / 2), 0, 0, np.sin(angle / 2)])
        obj_quat = np.zeros(4)
        mujoco.mju_mulQuat(obj_quat, yaw_quat, self._obj_rest_quat)

        addr = self._obj_qpos_addr
        self.data.qpos[addr : addr + 3] = [obj_x, obj_y, obj_z]
        self.data.qpos[addr + 3 : addr + 7] = obj_quat

        self._initial_obj_z = obj_z

        goal_x = cx + rng.uniform(-DEFAULT_CUBE_SPAWN_HALF_SIZE, DEFAULT_CUBE_SPAWN_HALF_SIZE)
        goal_y = cy + rng.uniform(-DEFAULT_CUBE_SPAWN_HALF_SIZE, DEFAULT_CUBE_SPAWN_HALF_SIZE)
        goal_z = self._obj_spawn_z + rng.uniform(0, DEFAULT_MAX_GOAL_HEIGHT)

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

        self._prev_target = _REST_QPOS.copy()

        mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def _get_current_qpos(self) -> np.ndarray:
        return np.array(
            [self.data.qpos[self.model.jnt_qposadr[jid]] for jid in self._joint_ids],
            dtype=np.float64,
        )

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)

        if self.control_mode == "pd_joint_pos":
            ctrl = action
        elif self.control_mode == "pd_joint_delta_pos":
            ctrl = np.clip(self._get_current_qpos() + action, self._ctrl_low, self._ctrl_high)
        else:
            self._prev_target = np.clip(self._prev_target + action, self._ctrl_low, self._ctrl_high)
            ctrl = self._prev_target

        self.data.ctrl[self._actuator_ids] = ctrl

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


class PickYCBLiftEnv(PickYCBEnv):
    """Pick-YCB variant where success requires lifting the object above a height threshold."""

    def _get_info(self) -> dict:
        info = super()._get_info()
        lift_height = info["lift_height"]
        is_grasped = info["is_grasped"]
        info["success"] = (lift_height > DEFAULT_LIFT_THRESHOLD) and (is_grasped > 0.5)
        return info

    def _compute_reward(self, info: dict) -> float:
        tcp_to_obj_dist = info["tcp_to_obj_dist"]
        reach_progress = 1.0 - float(np.tanh(5.0 * tcp_to_obj_dist))

        is_grasped = info["is_grasped"] > 0.5

        lift_height = max(info["lift_height"], 0.0)
        lift_progress = float(np.tanh(5.0 * lift_height)) if is_grasped else 0.0

        is_complete = info["success"]

        return compute_normalized_reward(
            reach_progress=reach_progress,
            is_grasped=is_grasped,
            task_progress=lift_progress,
            is_complete=is_complete,
        )
