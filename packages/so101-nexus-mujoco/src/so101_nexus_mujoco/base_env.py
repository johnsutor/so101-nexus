from __future__ import annotations

from typing import Any, Literal

import gymnasium
import mujoco
import numpy as np
from gymnasium import spaces

from so101_nexus_core.config import (
    SO101_JOINT_NAMES,
    ControlMode,
    EnvironmentConfig,
)

_REST_QPOS = np.array(EnvironmentConfig().robot.rest_qpos_rad, dtype=np.float64)


class SO101NexusMuJoCoBaseEnv(gymnasium.Env):
    """Shared MuJoCo base class for SO101-Nexus tasks."""

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 50}
    model: mujoco.MjModel
    data: mujoco.MjData
    config: EnvironmentConfig
    _obj_geom_id: int
    action_space: spaces.Box
    observation_space: spaces.Space
    _wrist_renderer: mujoco.Renderer | None
    _renderer: mujoco.Renderer | None
    _viewer: Any | None
    _VALID_CONTROL_MODES: set[str] = {
        "pd_joint_pos",
        "pd_joint_delta_pos",
        "pd_joint_target_delta_pos",
    }
    _N_SUBSTEPS = 10

    def _init_common(
        self,
        *,
        config: EnvironmentConfig,
        render_mode: str | None,
        camera_mode: Literal["state_only", "wrist"],
        control_mode: ControlMode,
        robot_init_qpos_noise: float,
    ) -> None:
        if control_mode not in self._VALID_CONTROL_MODES:
            valid = sorted(self._VALID_CONTROL_MODES)
            raise ValueError(f"control_mode must be one of {valid}, got {control_mode!r}")
        if camera_mode not in ("state_only", "wrist"):
            raise ValueError(f"camera_mode must be state_only|wrist, got {camera_mode!r}")
        if config.camera.width <= 0 or config.camera.height <= 0:
            raise ValueError(
                f"camera dimensions must be > 0, got {config.camera.width}x{config.camera.height}"
            )

        self.config = config
        self.control_mode = control_mode
        self.render_mode = render_mode
        self.camera_mode = camera_mode
        self.camera_width = config.camera.width
        self.camera_height = config.camera.height
        self.robot_init_qpos_noise = robot_init_qpos_noise

    def _finish_model_setup(self) -> None:
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
        self._gripper_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
        self._jaw_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "moving_jaw_so101_v1"
        )
        self._tcp_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")

        self._gripper_geom_ids = self._get_collision_geoms(self._gripper_body_id)
        self._jaw_geom_ids = self._get_collision_geoms(self._jaw_body_id)

        static_pad_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "static_finger_pad")
        moving_pad_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "moving_finger_pad")
        if static_pad_id >= 0:
            self._gripper_geom_ids.add(static_pad_id)
        if moving_pad_id >= 0:
            self._jaw_geom_ids.add(moving_pad_id)

        arm_dof_count = len(SO101_JOINT_NAMES) - 1
        self._arm_qvel_addrs = np.array(
            [self.model.jnt_dofadr[self._joint_ids[i]] for i in range(arm_dof_count)],
            dtype=np.int32,
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
                self.model,
                height=self.camera_height,
                width=self.camera_width,
            )
            self.observation_space = spaces.Dict(
                {
                    "state": spaces.Box(
                        low=-np.inf, high=np.inf, shape=(self._state_obs_size(),), dtype=np.float64
                    ),
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
                low=-np.inf, high=np.inf, shape=(self._state_obs_size(),), dtype=np.float64
            )

        self._renderer = None
        self._viewer = None

    def _reset_robot_joints(self, init_qpos: np.ndarray | None = None) -> None:
        """Reset arm joints to a target configuration with optional noise.

        Parameters
        ----------
        init_qpos : np.ndarray or None
            If provided, joints are set to this exact configuration (no noise).
            If None, joints are reset to the default rest pose with Gaussian noise
            scaled by ``robot_init_qpos_noise``.
        """
        target = init_qpos if init_qpos is not None else _REST_QPOS
        for i, jid in enumerate(self._joint_ids):
            qpos_addr = self.model.jnt_qposadr[jid]
            noise = (
                0.0
                if init_qpos is not None
                else self.np_random.uniform(-self.robot_init_qpos_noise, self.robot_init_qpos_noise)
            )
            self.data.qpos[qpos_addr] = target[i] + noise
        self.data.ctrl[self._actuator_ids] = np.clip(target, self._ctrl_low, self._ctrl_high)

    def _randomize_wrist_camera(self) -> None:
        """Randomize wrist camera pose and field of view for domain randomization.

        No-ops when ``camera_mode`` is not ``"wrist"``. Pitch, position, and FOV
        are sampled uniformly from the ranges defined in ``config.camera``.
        """
        if self.camera_mode != "wrist":
            return
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

        fov_lo, fov_hi = self.config.camera.wrist_fov_deg_range
        self.model.cam_fovy[self._wrist_cam_id] = self.np_random.uniform(fov_lo, fov_hi)

    def _get_collision_geoms(self, body_id: int) -> set[int]:
        """Return the set of collision geometry IDs attached to a body.

        Parameters
        ----------
        body_id : int
            MuJoCo body ID to query.

        Returns
        -------
        set[int]
            IDs of all geoms with non-zero ``contype`` attached to ``body_id``.
        """
        geom_ids = set()
        for i in range(self.model.ngeom):
            if self.model.geom_bodyid[i] == body_id and self.model.geom_contype[i] != 0:
                geom_ids.add(i)
        return geom_ids

    def _state_obs_size(self) -> int:
        """Return the dimensionality of the flat state observation vector.

        Default: tcp_pose(7) + is_grasped(1) + obj_pose(7) + tcp_to_obj(3) = 18.
        Subclasses that add or remove fields should override this method.
        """
        return 18

    def _get_tcp_pose(self) -> np.ndarray:
        """Return the tool-centre-point pose as a 7-vector [x, y, z, qw, qx, qy, qz]."""
        pos = self.data.site_xpos[self._tcp_site_id].copy()
        mat = self.data.site_xmat[self._tcp_site_id].reshape(3, 3)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, mat.flatten())
        return np.concatenate([pos, quat])

    def _is_grasping(self) -> float:
        """Return 1.0 if the gripper and jaw are both in contact with the target object.

        Uses ``config.robot.grasp_force_threshold`` to filter low-force contacts.

        Returns
        -------
        float
            1.0 when a two-sided grasp is detected, 0.0 otherwise.
        """
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
        """Return True if all arm joints are below the static velocity threshold.

        Uses ``config.robot.static_vel_threshold`` as the cutoff.
        """
        arm_vels = self.data.qvel[self._arm_qvel_addrs]
        return bool(np.all(np.abs(arm_vels) < 0.2))

    def _get_current_qpos(self) -> np.ndarray:
        """Return the current joint positions for all controlled joints."""
        return np.array(
            [self.data.qpos[self.model.jnt_qposadr[jid]] for jid in self._joint_ids],
            dtype=np.float64,
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray | dict[str, np.ndarray], dict]:
        super().reset(seed=seed, options=options)
        mujoco.mj_resetData(self.model, self.data)

        init_qpos: np.ndarray | None = None
        if options is not None:
            raw = options.get("init_qpos")
            if raw is not None:
                init_qpos = np.asarray(raw, dtype=np.float64)
                if init_qpos.shape != (len(self._joint_ids),):
                    raise ValueError(
                        f"init_qpos shape {init_qpos.shape} != expected ({len(self._joint_ids)},)"
                    )

        self._reset_robot_joints(init_qpos=init_qpos)
        self._task_reset()
        self._randomize_wrist_camera()

        self._prev_target = (init_qpos if init_qpos is not None else _REST_QPOS).copy()
        mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray | dict[str, np.ndarray], float, bool, bool, dict]:
        action = np.clip(action, self.action_space.low, self.action_space.high)

        if self.control_mode == "pd_joint_pos":
            ctrl = action
        elif self.control_mode == "pd_joint_delta_pos":
            ctrl = np.clip(self._get_current_qpos() + action, self._ctrl_low, self._ctrl_high)
        else:
            self._prev_target = np.clip(self._prev_target + action, self._ctrl_low, self._ctrl_high)
            ctrl = self._prev_target

        self.data.ctrl[self._actuator_ids] = ctrl

        for _ in range(self._N_SUBSTEPS):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        info = self._get_info()
        reward = self._compute_reward(info)
        terminated = bool(info.get("success", False))

        return obs, reward, terminated, False, info

    def render(self) -> np.ndarray | None:
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            self._renderer.update_scene(self.data)
            return self._renderer.render()
        if self.render_mode == "human":
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
            return None
        return None

    def close(self) -> None:
        """Release MuJoCo renderers and viewer resources."""
        if self._wrist_renderer is not None:
            self._wrist_renderer.close()
            self._wrist_renderer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    def _reach_only_reward(self, info: dict) -> float:
        """Reach-only reward: tanh distance shaping toward the object with no task progress."""
        reach_progress = 1.0 - float(np.tanh(5.0 * info["tcp_to_obj_dist"]))
        is_grasped = info["is_grasped"] > 0.5
        return self.config.reward.compute(
            reach_progress=reach_progress,
            is_grasped=is_grasped,
            task_progress=0.0,
            is_complete=info.get("success", False),
            action_delta_norm=info.get("action_delta_norm", 0.0),
        )

    def _lift_reward(self, info: dict) -> float:
        """Lift reward: reach + grasp + tanh lift shaping + completion bonus."""
        reach_progress = 1.0 - float(np.tanh(5.0 * info["tcp_to_obj_dist"]))
        is_grasped = info["is_grasped"] > 0.5
        lift_progress = float(np.tanh(5.0 * max(info["lift_height"], 0.0))) if is_grasped else 0.0
        return self.config.reward.compute(
            reach_progress=reach_progress,
            is_grasped=is_grasped,
            task_progress=lift_progress,
            is_complete=info.get("success", False),
            action_delta_norm=info.get("action_delta_norm", 0.0),
        )

    def _task_reset(self) -> None:
        raise NotImplementedError

    def _get_obs(self) -> np.ndarray | dict[str, np.ndarray]:
        raise NotImplementedError

    def _get_info(self) -> dict:
        raise NotImplementedError

    def _compute_reward(self, info: dict) -> float:
        raise NotImplementedError
