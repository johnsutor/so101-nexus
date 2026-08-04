"""MuJoCo base environment for SO101-Nexus simulation tasks."""

from __future__ import annotations

import logging
from enum import Enum
from functools import cache, wraps
from typing import TYPE_CHECKING, Any, Protocol

import gymnasium
import mujoco
import numpy as np
from gymnasium import spaces

from so101_nexus.camera_utils import compute_angled_camera_params, compute_overhead_camera_params
from so101_nexus.config import (
    EE_CONTROL_MODES,
    JOINT_CONTROL_MODES,
    SO101_ARM_JOINT_COUNT,
    SO101_JOINT_NAMES,
    SO101_TCP_SITE_NAME,
    ControlMode,
    EnvironmentConfig,
)
from so101_nexus.gaze import direction_to_object, gaze_angle_rad, gaze_cosine, object_in_view
from so101_nexus.grasp import opposing_normals_ok
from so101_nexus.kinematics import (
    EE_ACTION_DIM,
    EE_IK_ITERATIONS,
    ee_ik_delta_q,
    quat_multiply,
    rotvec_to_quat,
)
from so101_nexus.observations import (
    CameraObservation,
    EndEffectorPose,
    GazeDirection,
    GazeState,
    GraspState,
    GripperContactForce,
    JointEfforts,
    JointPositions,
    JointVelocities,
    ObjectOffset,
    ObjectPose,
    ObjectVelocity,
    OverheadCamera,
    TargetOffset,
    TargetPosition,
    WristCamera,
)
from so101_nexus.rewards import lift_progress, potential_shaping, reach_progress

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

# Internal per-joint physical scale applied to a normalized delta action.
# The public delta action space is normalized to [-1, 1] (the cross-backend
# normalized delta action contract); a normalized action ``a`` maps to a
# physical joint-target delta of ``a * _DELTA_ACTION_SCALE``. These are the
# existing controller delta units (radians): +/-0.05 for the five arm joints
# and +/-0.2 for the gripper. Reused by both delta control modes.
_DELTA_ACTION_SCALE = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.2], dtype=np.float64)

# The SO-101 arm has five actuated joints, so its tool Jacobian is rank 5: one
# twist direction is always unreachable and full SE(3) pose control is not
# achievable. Both end-effector modes still take a 6-DoF pose command and
# de-weight the orientation error by config.robot.ee_orientation_weight, so
# position tracks essentially exactly while orientation is best-effort. The Warp
# backend diverges from the nominal 6-DoF contract in exactly the same way.
#
# Half-width of the pd_ee_pose position box, in metres. Sampling the reachable
# set gives a maximum TCP reach of 0.5457 m, so 0.55 m spans the workspace
# without admitting targets the solver could only clamp toward.
_EE_WORKSPACE_RADIUS = 0.55


class _Dispatch(Enum):
    """The two observation branches that are not a base reader method.

    ``TASK`` routes the component through ``_get_component_data``; ``SKIP``
    marks a camera component, rendered in ``_get_obs`` rather than placed in
    the flat state vector. An enum rather than string constants so a mis-typed
    reader name can never be mistaken for a branch, which would silently
    reshape the observation vector instead of raising.
    """

    TASK = "task"
    SKIP = "camera"


_CACHE_MISS = object()


class _Reader[ReadT](Protocol):
    """An unbound zero-argument physics reader, as ``_observation_scoped`` sees it.

    Narrower than ``Callable`` because the decorator keys its memo on the
    method's ``__name__``.
    """

    __name__: str

    def __call__(self, env: Any, /) -> ReadT:
        """Read the quantity off the environment's live MuJoCo state."""
        ...


def _observation_scoped[ReadT](method: _Reader[ReadT]) -> Callable[[Any], ReadT]:
    """Memoize a zero-argument physics reader for the duration of one ``_observe``.

    The decorated readers are pure functions of the current ``self.data``, and
    ``_get_obs`` and ``_get_info`` each ask for several of the same ones while
    describing a single physics state. Outside the ``_observe`` window
    ``_read_cache`` is ``None`` and every call recomputes from live MuJoCo
    state, so no caller can be handed a value belonging to an earlier state.
    An instance attribute assigned over the reader (how the tests pin a grasp)
    shadows this wrapper entirely and is never cached.

    A memoized reader hands the same array to every caller inside the window,
    so a decorated reader must not return anything a caller mutates in place.
    """
    key = method.__name__

    @wraps(method)
    def reader(self: Any) -> ReadT:
        memo = self._read_cache
        if memo is None:
            return method(self)
        value = memo.get(key, _CACHE_MISS)
        if value is _CACHE_MISS:
            value = memo[key] = method(self)
        return value

    return reader


def _configure_free_camera(cam: mujoco.MjvCamera, params: dict[str, Any]) -> None:
    """Point a free camera at the lookat/distance/elevation/azimuth in ``params``."""
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = params["lookat"]
    cam.distance = params["distance"]
    cam.elevation = params["elevation"]
    cam.azimuth = params["azimuth"]


class SO101NexusMuJoCoBaseEnv(gymnasium.Env):
    """Shared MuJoCo base class for SO101-Nexus tasks.

    Notes
    -----
    ``_is_grasping()`` requires ``_obj_geom_id`` to be set by the subclass.
    Primitive envs without a graspable object (``LookAtEnv``, ``MoveEnv``)
    must **never** call ``_is_grasping()``.
    """

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 50}
    model: mujoco.MjModel
    data: mujoco.MjData
    config: EnvironmentConfig
    _obj_geom_id: int
    # Options dict of the current episode's reset(), for _task_reset to read.
    # Declared, never assigned at class level: a mutable class default would be
    # shared by every instance the moment someone mutates it in place.
    _reset_options: dict[str, Any]
    _target_index_consumed: bool
    action_space: spaces.Box
    observation_space: spaces.Space
    _wrist_renderer: mujoco.Renderer | None
    _renderer: mujoco.Renderer | None
    _viewer: Any | None
    _VALID_CONTROL_MODES: frozenset[str] = frozenset(JOINT_CONTROL_MODES + EE_CONTROL_MODES)
    # Menagerie physics uses timestep=0.005; keep control_dt = timestep *
    # _N_SUBSTEPS = 0.02 s (unchanged from the old 0.002 * 10).
    _N_SUBSTEPS = 4
    # Memo for the @_observation_scoped readers, non-None only inside _observe.
    # A plain class-level None is safe here (unlike a mutable default): _observe
    # rebinds it per instance for the window and clears it back to None.
    _read_cache: dict[str, Any] | None = None

    def _init_common(
        self,
        *,
        config: EnvironmentConfig,
        render_mode: str | None,
        control_mode: ControlMode,
        robot_init_qpos_noise: float,
    ) -> None:
        if control_mode not in self._VALID_CONTROL_MODES:
            valid = sorted(self._VALID_CONTROL_MODES)
            raise ValueError(f"control_mode must be one of {valid}, got {control_mode!r}")

        self.config = config
        self.control_mode = control_mode
        self.render_mode = render_mode
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self._privileged_state: np.ndarray | None = None
        self._init_qpos_clamp_warned = False

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
        self._tcp_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, SO101_TCP_SITE_NAME
        )
        # The wrist camera is part of the robot model, so its pose and FOV are
        # defined whether or not a WristCamera observation renders from it: the
        # gaze components and the look-at success predicate read them directly.
        # mj_name2id's -1 sentinel would silently address the last camera, so it
        # is rejected here, matching the Warp backend's lookup.
        self._wrist_cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
        if self._wrist_cam_id < 0:
            raise RuntimeError("the scene's robot model does not define a 'wrist_cam' camera")

        # Menagerie finger contact surfaces use condim=6 (collision_gripper and
        # collision_gripper_mesh classes); the non-finger wrist-roll box on the
        # gripper body uses condim=3 and is excluded. Camera-mount boxes live on
        # the separate camera_mount child body and are excluded by the per-body
        # filter in _get_finger_geoms.
        self._gripper_geom_ids = self._get_finger_geoms(self._gripper_body_id)
        self._jaw_geom_ids = self._get_finger_geoms(self._jaw_body_id)
        # Same two sets as a per-geom boolean lookup, so the contact scan can
        # classify every contact with one fancy-index instead of a Python
        # membership test per contact.
        self._finger_geom_mask = np.zeros(self.model.ngeom, dtype=bool)
        self._finger_geom_mask[list(self._gripper_geom_ids | self._jaw_geom_ids)] = True

        # Per-joint qpos/qvel addresses for the six controlled joints; the arm
        # slices (gripper excluded) back the static-robot check and IK.
        self._qpos_addrs = np.array(
            [self.model.jnt_qposadr[jid] for jid in self._joint_ids], dtype=np.int32
        )
        self._qvel_addrs = np.array(
            [self.model.jnt_dofadr[jid] for jid in self._joint_ids], dtype=np.int32
        )
        self._arm_qpos_addrs = self._qpos_addrs[:SO101_ARM_JOINT_COUNT]
        self._arm_qvel_addrs = self._qvel_addrs[:SO101_ARM_JOINT_COUNT]

        ctrl_range = self.model.actuator_ctrlrange[self._actuator_ids]
        ctrl_low = ctrl_range[:, 0]
        ctrl_high = ctrl_range[:, 1]

        # Valid position-target bounds: a commanded actuator target must lie
        # within both the actuator ctrlrange and the compiled joint range. The
        # menagerie wrist_roll actuator advertises a wider ctrlrange than its
        # joint limit, so commanding the ctrlrange edge would drive the joint
        # into its limit. Clamp every position target (reset and control) to the
        # intersection.
        jnt_range = self.model.jnt_range[self._joint_ids]
        self._target_low = np.maximum(ctrl_low, jnt_range[:, 0])
        self._target_high = np.minimum(ctrl_high, jnt_range[:, 1])

        if self.control_mode == "pd_joint_pos":
            self.action_space = spaces.Box(
                low=self._target_low.astype(np.float32),
                high=self._target_high.astype(np.float32),
                dtype=np.float32,
            )
        elif self.control_mode == "pd_ee_pose":
            # Absolute TCP pose: world position, orientation as a rotation
            # vector, then the gripper joint target on pd_joint_pos's bounds.
            ee_low = np.concatenate(
                [np.full(3, -_EE_WORKSPACE_RADIUS), np.full(3, -np.pi), self._target_low[-1:]]
            )
            ee_high = np.concatenate(
                [np.full(3, _EE_WORKSPACE_RADIUS), np.full(3, np.pi), self._target_high[-1:]]
            )
            self.action_space = spaces.Box(
                low=ee_low.astype(np.float32),
                high=ee_high.astype(np.float32),
                dtype=np.float32,
            )
        else:
            # Delta modes expose a normalized [-1, 1] action space (the
            # cross-backend contract). A normalized action is scaled in step()
            # before it reaches the targets: joint modes by _DELTA_ACTION_SCALE,
            # the end-effector mode by the robot config's ee delta scale.
            n_actions = (
                EE_ACTION_DIM if self.control_mode == "pd_ee_delta_pose" else len(SO101_JOINT_NAMES)
            )
            self.action_space = spaces.Box(
                low=-np.ones(n_actions, dtype=np.float32),
                high=np.ones(n_actions, dtype=np.float32),
                dtype=np.float32,
            )

        # Scratch state so the IK loop can re-evaluate forward kinematics
        # without perturbing the live simulation.
        self._ik_data: mujoco.MjData | None = (
            mujoco.MjData(self.model) if self.control_mode in EE_CONTROL_MODES else None
        )
        self._ee_delta_scale = np.asarray(self.config.robot.ee_delta_action_scale, dtype=np.float64)

        self._prev_target: np.ndarray | None = None
        # Previous public policy action, used for the action-smoothness penalty.
        # None means "no action since the last reset" so the first step reports
        # an action_delta_norm of 0.0.
        self._prev_action: np.ndarray | None = None
        self._setup_camera_renderers()
        self._renderer = None
        self._render_cam: mujoco.MjvCamera | None = None
        self._viewer = None

    def _setup_camera_renderers(self) -> None:
        """Detect camera observation components and set up renderers + obs space."""
        self._wrist_cam_component: WristCamera | None = None
        self._overhead_cam_component: OverheadCamera | None = None
        if self.config.observations is not None:
            for comp in self.config.observations:
                if isinstance(comp, WristCamera):
                    self._wrist_cam_component = comp
                elif isinstance(comp, OverheadCamera):
                    self._overhead_cam_component = comp

        if self._wrist_cam_component is not None:
            wrist_w = self._wrist_cam_component.width
            wrist_h = self._wrist_cam_component.height
            self._wrist_renderer = mujoco.Renderer(self.model, height=wrist_h, width=wrist_w)
        else:
            self._wrist_renderer = None

        if self._overhead_cam_component is not None:
            cam = self._overhead_cam_component
            params = compute_overhead_camera_params(
                spawn_center=self.config.spawn_center,
                spawn_max_radius=self.config.spawn_max_radius,
                fov_deg=cam.fov_deg,
                aspect=cam.width / cam.height,
            )
            self._overhead_obs_cam = mujoco.MjvCamera()
            _configure_free_camera(self._overhead_obs_cam, params)
            self._overhead_obs_renderer = mujoco.Renderer(
                self.model, height=cam.height, width=cam.width
            )
        else:
            self._overhead_obs_cam = None
            self._overhead_obs_renderer = None

        self.observation_space = self._build_observation_space()

    def _build_observation_space(self) -> spaces.Space:
        """Build observation space from detected camera components."""
        has_any_camera = (
            self._wrist_cam_component is not None or self._overhead_cam_component is not None
        )
        if not has_any_camera:
            return spaces.Box(
                low=-np.inf, high=np.inf, shape=(self._state_obs_size(),), dtype=np.float32
            )
        state_size = (
            len(SO101_JOINT_NAMES) if self.config.obs_mode == "visual" else self._state_obs_size()
        )
        obs_dict: dict[str, spaces.Space] = {
            "state": spaces.Box(low=-np.inf, high=np.inf, shape=(state_size,), dtype=np.float32),
        }
        if self._wrist_cam_component is not None:
            wc = self._wrist_cam_component
            obs_dict["wrist_camera"] = spaces.Box(
                low=0,
                high=255,
                shape=(wc.height, wc.width, 3),
                dtype=np.uint8,
            )
        if self._overhead_cam_component is not None:
            oc = self._overhead_cam_component
            obs_dict["overhead_camera"] = spaces.Box(
                low=0,
                high=255,
                shape=(oc.height, oc.width, 3),
                dtype=np.uint8,
            )
        return spaces.Dict(obs_dict)

    def _reset_robot_joints(self, init_qpos: np.ndarray | None = None) -> np.ndarray:
        """Reset arm joints to a target configuration with optional noise.

        Parameters
        ----------
        init_qpos : np.ndarray or None
            If provided, joints are set to this configuration clipped to
            actuator control bounds (no noise).
            If None, joints are reset based on ``config.robot.init_pose`` if set,
            otherwise the default rest pose with Gaussian noise.

        Returns
        -------
        np.ndarray
            The target joint positions actually applied (before per-joint noise).
        """
        if init_qpos is not None:
            target = np.asarray(init_qpos, dtype=np.float64).copy()
            clipped = np.clip(target, self._target_low, self._target_high)
            if not np.array_equal(target, clipped):
                if not self._init_qpos_clamp_warned:
                    logger.warning(
                        "init_qpos clipped to control bounds for at least one joint; "
                        "further init_qpos clamps on this env will be silent."
                    )
                    self._init_qpos_clamp_warned = True
                target = clipped
            noise_scale = 0.0
        else:
            pose = self.config.robot.resolve_pose()
            if pose is not None:
                target = np.array(pose.sample_rad(self.np_random), dtype=np.float64)
                noise_scale = 0.0  # free joints already have randomness
            else:
                target = np.array(self.config.robot.rest_qpos_rad, dtype=np.float64)
                noise_scale = self.robot_init_qpos_noise

        # Clamp the position target to the valid bounds so the held ctrl target
        # (and the returned value seeding _prev_target) never commands past the
        # joint limit; a zero action in target-delta mode must hold this pose.
        applied = np.clip(target, self._target_low, self._target_high)
        for i, jid in enumerate(self._joint_ids):
            qpos_addr = self.model.jnt_qposadr[jid]
            noise = 0.0 if noise_scale == 0.0 else self.np_random.uniform(-noise_scale, noise_scale)
            # Re-clamp after adding noise so noise cannot push a joint past its limit.
            self.data.qpos[qpos_addr] = np.clip(
                applied[i] + noise, self._target_low[i], self._target_high[i]
            )
        self.data.ctrl[self._actuator_ids] = applied
        return applied

    def _randomize_wrist_camera(self) -> None:
        """Randomize wrist camera pose and field of view for domain randomization.

        Uses ``WristCamera`` observation component parameters. No-ops when no
        wrist camera is active.
        """
        if self._wrist_renderer is None:
            return

        wc = self._wrist_cam_component
        if wc is None:
            raise RuntimeError("wrist renderer requires a WristCamera component")
        pitch_lo_rad, pitch_hi_rad = wc.pitch_rad_range
        fov_lo, fov_hi = wc.fov_deg_range
        pos_x_noise = wc.pos_x_noise
        pos_y_center, pos_y_noise = wc.pos_y_center, wc.pos_y_noise
        pos_z_center, pos_z_noise = wc.pos_z_center, wc.pos_z_noise

        # Write the body-relative pose fields the fixed-camera forward kinematics
        # actually use (cam_pos / cam_quat). The earlier cam_pos0 / cam_mat0
        # fields are only consulted for tracking-mode cameras, so writing them
        # left this fixed wrist camera at its MJCF baseline pose (the pitch and
        # position randomization had no effect). The reset's mj_forward (called
        # right after this) recomputes cam_xpos / cam_xmat from these fields.
        pitch_rad = self.np_random.uniform(pitch_lo_rad, pitch_hi_rad)
        quat = np.zeros(4)
        mujoco.mju_euler2Quat(quat, np.array([pitch_rad, 0.0, 0.0]), "XYZ")
        self.model.cam_quat[self._wrist_cam_id] = quat

        self.model.cam_pos[self._wrist_cam_id] = [
            self.np_random.uniform(-pos_x_noise, pos_x_noise),
            pos_y_center + self.np_random.uniform(-pos_y_noise, pos_y_noise),
            pos_z_center + self.np_random.uniform(-pos_z_noise, pos_z_noise),
        ]

        self.model.cam_fovy[self._wrist_cam_id] = self.np_random.uniform(fov_lo, fov_hi)

    def _get_finger_geoms(self, body_id: int) -> set[int]:
        """Return the ``condim==6`` collision geom IDs attached to a finger body.

        Selects the menagerie gripper contact surfaces (both ``collision_gripper``
        primitives and ``collision_gripper_mesh`` meshes, which use ``condim=6``)
        while excluding the non-finger wrist-roll follower box on the gripper body
        (plain ``collision`` class, ``condim=3``).

        Parameters
        ----------
        body_id : int
            MuJoCo body ID of the finger (gripper or moving jaw) to query.

        Returns
        -------
        set[int]
            IDs of contype-enabled, ``condim==6`` geoms attached to ``body_id``.
        """
        return {
            i
            for i in range(self.model.ngeom)
            if self.model.geom_bodyid[i] == body_id
            and self.model.geom_contype[i] != 0
            and self.model.geom_condim[i] == 6
        }

    def _state_obs_size(self) -> int:
        """Return the dimensionality of the flat state observation vector."""
        if self.config.observations is not None:
            return sum(c.size for c in self.config.observations if c.size > 0)
        # Legacy default for pick envs that haven't migrated yet
        return 18

    @_observation_scoped
    def _get_tcp_pose(self) -> np.ndarray:
        """Return the tool-centre-point pose as a 7-vector [x, y, z, qw, qx, qy, qz]."""
        pos = self.data.site_xpos[self._tcp_site_id].copy()
        mat = self.data.site_xmat[self._tcp_site_id].reshape(3, 3)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, mat.flatten())
        return np.concatenate([pos, quat])

    def _solve_ee_ik(
        self, target_pos: np.ndarray, target_quat: np.ndarray, gripper_target: float
    ) -> np.ndarray:
        """Damped-least-squares joint targets tracking a commanded TCP pose.

        Parameters
        ----------
        target_pos : np.ndarray
            Target TCP position in world metres, shape ``(3,)``.
        target_quat : np.ndarray
            Target TCP orientation as a ``[w, x, y, z]`` quaternion, shape ``(4,)``.
        gripper_target : float
            Gripper joint target in radians.

        Returns
        -------
        np.ndarray
            Six actuator position targets (five arm joints then the gripper),
            each clamped to the valid target bounds. Targets outside the
            reachable workspace resolve to the closest achievable pose rather
            than raising.
        """
        ik_data = self._ik_data
        assert ik_data is not None  # allocated for EE modes in _finish_model_setup

        # Scene objects carry free joints, so seed the whole configuration; the
        # arm occupies only the first six qpos entries.
        ik_data.qpos[:] = self.data.qpos
        arm_qpos_addrs = self._arm_qpos_addrs
        arm_dofs = self._arm_qvel_addrs
        q = self.data.qpos[arm_qpos_addrs].copy()

        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        current_quat = np.zeros(4)
        for _ in range(EE_IK_ITERATIONS):
            ik_data.qpos[arm_qpos_addrs] = q
            # mj_jacSite reads subtree centre-of-mass data, so mj_kinematics
            # alone leaves the rotational block stale; mj_comPos refreshes it.
            mujoco.mj_kinematics(self.model, ik_data)
            mujoco.mj_comPos(self.model, ik_data)
            mujoco.mj_jacSite(self.model, ik_data, jacp, jacr, self._tcp_site_id)
            mujoco.mju_mat2Quat(current_quat, ik_data.site_xmat[self._tcp_site_id])
            # ee_ik_delta_q applies the orientation weight to the rotational
            # rows itself, so this Jacobian is handed over raw.
            dq = ee_ik_delta_q(
                np.vstack([jacp[:, arm_dofs], jacr[:, arm_dofs]]),
                ik_data.site_xpos[self._tcp_site_id],
                current_quat,
                target_pos,
                target_quat,
                orientation_weight=self.config.robot.ee_orientation_weight,
            )
            q = np.clip(
                q + dq,
                self._target_low[:SO101_ARM_JOINT_COUNT],
                self._target_high[:SO101_ARM_JOINT_COUNT],
            )

        gripper = np.clip(gripper_target, self._target_low[-1], self._target_high[-1])
        return np.concatenate([q, [gripper]])

    @_observation_scoped
    def _is_grasping(self) -> float:
        """Return 1.0 when the two finger sets pinch the target object.

        Both finger sets must contact the target with normal force at or above
        ``config.robot.grasp_force_threshold``, and the force-weighted mean
        contact normals of the two sides must oppose each other by at least
        ``config.robot.grasp_opposing_normal_threshold`` (see
        ``so101_nexus.grasp.opposing_normals_ok``). The opposition term is what
        rejects a straddle: an object too wide for the jaw to close on is
        touched by both finger sets from the same side while it rests on the
        table, which satisfies bilateral contact but bears no load.

        Returns
        -------
        float
            1.0 when a pinching two-sided grasp is detected, 0.0 otherwise.
        """
        # Building a per-contact wrapper (data.contact[i]) costs more than the
        # force solve itself, so the object's contacts are selected from the
        # struct-of-arrays view first; ascending order matches the old scan, so
        # the accumulation is unchanged. _obj_geom_id is read only once there is
        # a contact to classify, matching the old scan for the primitive envs
        # that never define it.
        contacts = self.data.contact
        geoms = contacts.geom[: self.data.ncon]
        if geoms.shape[0] == 0:
            return 0.0
        obj_geom_id = self._obj_geom_id
        rows = np.flatnonzero((geoms == obj_geom_id).any(axis=1))
        if rows.size == 0:
            return 0.0

        gripper_normal = np.zeros(3)
        jaw_normal = np.zeros(3)
        frames = contacts.frame
        force_buf = np.zeros(6)
        for i in rows:
            g1, g2 = geoms[i]
            other = int(g2 if g1 == obj_geom_id else g1)
            if other in self._gripper_geom_ids:
                accum = gripper_normal
            elif other in self._jaw_geom_ids:
                accum = jaw_normal
            else:
                continue

            mujoco.mj_contactForce(self.model, self.data, i, force_buf)
            normal_force = abs(force_buf[0])
            if normal_force < self.config.robot.grasp_force_threshold:
                continue

            # The contact frame's first row is the normal pointing from geom1 to
            # geom2; flip it when the object is geom1 so it points into the
            # object for both orderings.
            normal = frames[i, :3] if g2 == obj_geom_id else -frames[i, :3]
            accum += normal_force * normal

        if not (gripper_normal.any() and jaw_normal.any()):
            return 0.0
        return float(
            opposing_normals_ok(
                gripper_normal,
                jaw_normal,
                threshold=self.config.robot.grasp_opposing_normal_threshold,
            )
        )

    def _is_robot_static(self) -> bool:
        """Return True if all arm joints are below the static velocity threshold.

        Uses ``config.robot.static_vel_threshold`` as the cutoff.
        """
        arm_vels = self.data.qvel[self._arm_qvel_addrs]
        return bool(np.all(np.abs(arm_vels) < self.config.robot.static_vel_threshold))

    def _gaze_target_pos(self) -> np.ndarray:
        """Return the world position of the object the task acts on.

        Implemented by every task that has one; the gaze components and the
        look-at predicate are undefined without it (``MoveEnv`` has no object).
        """
        raise NotImplementedError(
            f"{type(self).__name__} has no target object, so it has no gaze target"
        )

    def _gaze_axis(self) -> np.ndarray:
        """Return the wrist-camera optical axis in world frame (where it points)."""
        # MuJoCo cameras look along their local -z axis, so the optical axis is
        # the negated third column of the camera rotation matrix. Using the real
        # camera axis (not a gripper-frame proxy) keeps the gaze tied to what the
        # camera sees, and tracks any mount/FOV randomization automatically.
        return -self.data.cam_xmat[self._wrist_cam_id].reshape(3, 3)[:, 2].copy()

    def _gaze_direction(self) -> np.ndarray:
        """Return the unit vector from the wrist camera toward the target object."""
        return direction_to_object(self.data.cam_xpos[self._wrist_cam_id], self._gaze_target_pos())

    def _gaze_cosine(self) -> float:
        """Return the cosine between the optical axis and the target object."""
        return float(gaze_cosine(self._gaze_axis(), self._gaze_direction()))

    def _gaze_angle_rad(self) -> float:
        """Return the angle between the optical axis and the target object."""
        return float(gaze_angle_rad(self._gaze_cosine()))

    def _half_fov_rad(self) -> float:
        """Half the live wrist-camera vertical FOV (radians): the in-frame boundary."""
        return float(np.radians(self.model.cam_fovy[self._wrist_cam_id].item()) / 2.0)

    def _is_looking_at(self) -> float:
        """Return 1.0 when the target object is inside the wrist camera's FOV."""
        return float(object_in_view(self._gaze_angle_rad(), self._half_fov_rad()))

    @property
    def control_dt(self) -> float:
        """Simulated seconds advanced by one ``step()`` (physics timestep x substeps).

        This is the time between consecutive observations, and therefore the
        correct ``dt`` for finite-differencing recorded joint positions. It is
        unrelated to a teleop recording's wall-clock fps: the recorder sleeps to
        pace the operator but advances the simulation exactly one step per frame.
        """
        return float(self.model.opt.timestep) * self._N_SUBSTEPS

    def _get_current_qpos(self) -> np.ndarray:
        """Return the current joint positions for all controlled joints."""
        return self.data.qpos[self._qpos_addrs]

    def _get_current_qvel(self) -> np.ndarray:
        """Return the current joint velocities (rad/s) for all controlled joints."""
        return self.data.qvel[self._qvel_addrs]

    def _get_current_qfrc_actuator(self) -> np.ndarray:
        """Return the actuator generalized force (N*m) for all controlled joints."""
        return self.data.qfrc_actuator[self._qvel_addrs]

    def _get_gripper_contact_force(self) -> np.ndarray:
        """Return the world-frame resultant contact force applied to the fingers.

        Sums every contact involving a finger contact geom, with the sign chosen
        so the result is the force acting *on* the gripper. Unlike
        ``_is_grasping`` this needs no target object, so primitive envs can use
        it too.
        """
        # Same struct-of-arrays prefilter as _is_grasping: build the per-contact
        # wrapper only for the contacts that turn out to involve a finger.
        contacts = self.data.contact
        geoms = contacts.geom[: self.data.ncon]
        fingers = self._finger_geom_mask[geoms]
        rows = np.flatnonzero(fingers[:, 0] != fingers[:, 1])
        total = np.zeros(3)
        if rows.size == 0:
            return total

        frames = contacts.frame
        force_buf = np.zeros(6)
        for i in rows:
            mujoco.mj_contactForce(self.model, self.data, i, force_buf)
            # mj_contactForce reports the force on geom2 in the contact frame,
            # whose rows are the normal and the two tangents.
            world = frames[i].reshape(3, 3).T @ force_buf[:3]
            total += world if fingers[i, 1] else -world
        return total

    #: Components the base reads directly, by reader method name. Resolved
    #: through the MRO so a user subclass of a component behaves like the
    #: component it derives from, matching the ``isinstance`` dispatch used for
    #: the remaining branches. The gaze readers are here rather than per task
    #: because only the target position they resolve is task-specific (see
    #: ``_gaze_target_pos``). Everything else is either a camera or a task
    #: component routed through ``_get_component_data``.
    _BASE_COMPONENT_READERS: dict[type, str] = {
        JointPositions: "_get_current_qpos",
        JointVelocities: "_get_current_qvel",
        JointEfforts: "_get_current_qfrc_actuator",
        GripperContactForce: "_get_gripper_contact_force",
        EndEffectorPose: "_get_tcp_pose",
        GazeDirection: "_gaze_direction",
    }

    #: Task components routed to ``_get_component_data``; a component outside
    #: both this tuple and ``_BASE_COMPONENT_READERS`` is rejected.
    _TASK_COMPONENTS: tuple[type, ...] = (
        TargetOffset,
        ObjectPose,
        ObjectVelocity,
        ObjectOffset,
        TargetPosition,
    )

    def _grasp_state_obs(self) -> np.ndarray:
        """GraspState as a 1-vector."""
        return np.array([self._is_grasping()])

    def _gaze_state_obs(self) -> np.ndarray:
        """GazeState as a 1-vector."""
        return np.array([self._is_looking_at()])

    def _compute_obs_components(self) -> np.ndarray:
        """Build the flat state vector from the observation component list."""
        parts: list[np.ndarray] = []
        if self.config.observations is None:
            raise RuntimeError("config.observations must be set")
        for comp in self.config.observations:
            reader = self._component_reader(type(comp))
            if reader is _Dispatch.SKIP:
                continue  # camera images handled separately in _get_obs
            if reader is _Dispatch.TASK:
                parts.append(self._get_component_data(comp))
            elif reader is None:
                raise ValueError(f"Unsupported observation component: {comp!r}")
            else:
                parts.append(getattr(self, reader)())
        return np.concatenate(parts).astype(np.float32, copy=False)

    @classmethod
    @cache
    def _component_reader(cls, component_type: type) -> str | _Dispatch | None:
        """Resolve a component type to its dispatch branch, once per (class, type).

        Returns a base reader method name, ``_Dispatch.TASK`` for components
        ``_get_component_data`` owns, ``_Dispatch.SKIP`` for camera components,
        or ``None`` when the component is unsupported. Resolution is MRO-based,
        so a user subclass of a component dispatches like its base, matching the
        ``isinstance`` chain this replaces. Caching it keeps the per-step
        observation build off ``ABCMeta.__instancecheck__``, which dominated the
        dispatch cost for a ten-component observation list. The cache is keyed on
        the class, so a subclass may rebind ``_BASE_COMPONENT_READERS`` or
        ``_TASK_COMPONENTS``, but must never mutate either in place: the first
        lookup for a component type is the one that sticks.
        """
        readers = cls._BASE_COMPONENT_READERS
        for klass in component_type.__mro__:
            if klass in readers:
                return readers[klass]
        if issubclass(component_type, GraspState):
            return "_grasp_state_obs"
        if issubclass(component_type, GazeState):
            return "_gaze_state_obs"
        if issubclass(component_type, cls._TASK_COMPONENTS):
            return _Dispatch.TASK
        if issubclass(component_type, CameraObservation):
            return _Dispatch.SKIP
        return None

    def _resolve_target_index(self, n_pool: int) -> int | None:
        """Return this reset's ``options['target_index']`` override, or ``None``.

        Tasks that pick a target object out of a pool call this from
        ``_task_reset`` so a caller can hold the scene layout fixed and vary only
        the target, which one seeded RNG draw cannot express. Calling it is what
        marks the option consumed; ``reset`` rejects a ``target_index`` no task
        consumed, so a pin aimed at a poolless task fails loudly instead of
        silently collecting mislabelled data (matching the Warp backend).
        """
        raw = self._reset_options.get("target_index")
        if raw is None:
            return None
        self._target_index_consumed = True
        if isinstance(raw, bool) or not isinstance(raw, (int, np.integer)):
            raise ValueError(f"target_index must be an integer, got {raw!r}")
        index = int(raw)
        if not 0 <= index < n_pool:
            raise ValueError(
                f"target_index must be in [0, {n_pool}) for this object pool, got {raw!r}"
            )
        return index

    def _get_component_data(self, component: object) -> np.ndarray:
        """Return data for a task-specific observation component.

        Subclasses override this for components like TargetOffset or GazeDirection
        that depend on task state (target position, object position, etc.).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support observation component {component!r}"
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray | dict[str, np.ndarray], dict]:
        """Reset the environment and return the initial observation and info.

        Recognised ``options`` keys: ``init_qpos`` (explicit reset joint angles)
        and any task-specific key a subclass reads from ``_reset_options``
        during ``_task_reset`` (``target_index`` on the pick tasks). A
        ``target_index`` that the task never consumes raises, so a pin aimed at
        a task with no object pool cannot silently mislabel collected data.
        """
        super().reset(seed=seed, options=options)
        self._reset_options = {} if options is None else dict(options)
        self._target_index_consumed = False
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

        self._prev_action = None
        applied_qpos = self._reset_robot_joints(init_qpos=init_qpos)
        self._task_reset()
        if "target_index" in self._reset_options and not self._target_index_consumed:
            raise ValueError(f"{type(self).__name__} has no object pool to target")
        self._randomize_wrist_camera()

        self._prev_target = applied_qpos.copy()
        mujoco.mj_forward(self.model, self.data)
        self._settle_after_reset()
        self._refresh_reset_reference_state()

        return self._observe()

    def _observe(self) -> tuple[np.ndarray | dict[str, np.ndarray], dict]:
        """Return ``(obs, info)`` describing the current physics state.

        The two halves are separate methods but describe one state, and they
        ask the ``@_observation_scoped`` readers for several of the same
        quantities (a ten-component observation list plus a task info dict
        reads the TCP pose three times and the grasp predicate twice). This
        opens the memo window around both, and closes it on the way out so
        every read taken outside ``reset``/``step`` still hits live MuJoCo.
        """
        self._read_cache = {}
        try:
            return self._get_obs(), self._get_info()
        finally:
            self._read_cache = None

    def _settle_after_reset(self) -> None:
        """Advance configured no-op frames after reset before returning observations."""
        # data.ctrl was set during robot reset and must remain held while settling.
        for _ in range(self.config.reset_settle_frames):
            for _ in range(self._N_SUBSTEPS):
                mujoco.mj_step(self.model, self.data)

    def _refresh_reset_reference_state(self) -> None:
        """Refresh task reference state after reset settling."""

    def _action_to_ctrl(self, action: np.ndarray) -> np.ndarray:
        """Resolve a public action into six actuator position targets.

        Shared seam with the Warp backend's ``_action_to_ctrl``: it clips the
        action to the action space, applies the control mode's semantics and
        returns joint targets without stepping physics. ``pd_joint_target_delta_pos``
        advances its held target here, because integrating that target is the mode.

        Parameters
        ----------
        action : np.ndarray
            Public action shaped like this env's ``action_space``.

        Returns
        -------
        np.ndarray
            Six actuator position targets (five arm joints then the gripper).
        """
        action = np.clip(action, self.action_space.low, self.action_space.high)

        if self.control_mode == "pd_joint_pos":
            # action_space is already the valid target range for this mode.
            return action
        if self.control_mode == "pd_joint_delta_pos":
            # Normalized action in [-1, 1] is scaled to a physical joint delta.
            delta = action * _DELTA_ACTION_SCALE
            return np.clip(self._get_current_qpos() + delta, self._target_low, self._target_high)
        if self.control_mode == "pd_joint_target_delta_pos":
            # Normalized action in [-1, 1] is scaled to a physical joint delta.
            delta = action * _DELTA_ACTION_SCALE
            target = np.clip(self._prev_target + delta, self._target_low, self._target_high)
            self._prev_target = target
            return target

        ee_action = np.asarray(action, dtype=np.float64)
        if self.control_mode == "pd_ee_pose":
            return self._solve_ee_ik(
                ee_action[:3], rotvec_to_quat(ee_action[3:6]), float(ee_action[6])
            )

        # pd_ee_delta_pose integrates from the measured TCP pose, mirroring
        # pd_joint_delta_pos. The rotation delta is a world-frame twist, so it
        # left-multiplies the current TCP orientation.
        delta = ee_action * self._ee_delta_scale
        tcp = self._get_tcp_pose()
        return self._solve_ee_ik(
            tcp[:3] + delta[:3],
            quat_multiply(rotvec_to_quat(delta[3:6]), tcp[3:]),
            float(self._get_current_qpos()[-1] + delta[6]),
        )

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray | dict[str, np.ndarray], float, bool, bool, dict]:
        """Apply action, advance physics, and return (obs, reward, terminated, truncated, info)."""
        # Penalty norms use the public action as received here, before clipping,
        # following the cross-backend convention so the penalty is comparable
        # across backends. Clipping in _action_to_ctrl only affects the control
        # sent to physics.
        public_action = np.asarray(action, dtype=np.float64)
        self.data.ctrl[self._actuator_ids] = self._action_to_ctrl(action)

        for _ in range(self._N_SUBSTEPS):
            mujoco.mj_step(self.model, self.data)

        obs, info = self._observe()
        info["energy_norm"] = float(np.linalg.norm(public_action))
        if self._prev_action is None:
            info["action_delta_norm"] = 0.0
        else:
            info["action_delta_norm"] = float(np.linalg.norm(public_action - self._prev_action))
        self._prev_action = public_action.copy()
        reward = self._compute_reward(info)
        terminated = self.config.terminate_on_success and bool(info.get("success", False))

        return obs, reward, terminated, False, info

    def _render_camera_params(self) -> dict[str, Any]:
        """Free-camera params for the configured render view (overhead or side)."""
        render = self.config.render
        if render.camera == "side":
            return compute_angled_camera_params(
                spawn_center=self.config.spawn_center,
                spawn_max_radius=self.config.spawn_max_radius,
                elevation=render.side_elevation_deg,
                azimuth=render.side_azimuth_deg,
                aspect=render.width / render.height,
            )
        return compute_overhead_camera_params(
            spawn_center=self.config.spawn_center,
            spawn_max_radius=self.config.spawn_max_radius,
            aspect=render.width / render.height,
        )

    def render(self) -> np.ndarray | None:
        """Render the current frame and return an RGB array, or None."""
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(
                    self.model,
                    height=self.config.render.height,
                    width=self.config.render.width,
                )
            if self._render_cam is None:
                self._render_cam = mujoco.MjvCamera()
                _configure_free_camera(self._render_cam, self._render_camera_params())
            self._renderer.update_scene(self.data, camera=self._render_cam)
            return self._renderer.render()
        if self.render_mode == "human":
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
                # Open on the configured view; the user can still orbit freely.
                _configure_free_camera(self._viewer.cam, self._render_camera_params())
            self._viewer.sync()
            return None
        return None

    def close(self) -> None:
        """Release MuJoCo renderers and viewer resources."""
        if self._wrist_renderer is not None:
            self._wrist_renderer.close()
            self._wrist_renderer = None
        overhead_renderer = self._overhead_obs_renderer
        if overhead_renderer is not None:
            overhead_renderer.close()
            self._overhead_obs_renderer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    def _lift_reward(self, info: dict) -> float:
        """Lift reward: potential-shaped reach + grasp + lift + completion bonus.

        Stores the per-facet breakdown on ``info["reward_components"]`` (see
        ``RewardConfig.compute_components``) so recorders can persist each
        facet alongside the summed total returned here. ``reaching``,
        ``grasping``, and ``task_progress`` are all potential-based deltas (Ng,
        Harada & Russell, ICML 1999; see ``rewards.potential_shaping``), not raw
        state values -- each is a strict subset of ``success``'s completion
        surface (reach and grasp must happen before lift), so a raw (dwelling)
        value lets a policy park at "reached and grasped, never lifted" and
        collect up to their combined budget every step forever. Requires the
        caller to maintain ``self._prev_reach_progress``,
        ``self._prev_grasp_progress``, and ``self._prev_task_potential``, seeded
        post-settle by ``_refresh_reset_reference_state`` (see ``PickEnv``).
        """
        scale = self.config.reward.tanh_shaping_scale
        reach_now = reach_progress(info["tcp_to_obj_dist"], scale=scale)
        grasp_now = float(info["is_grasped"] > 0.5)
        reach_delta = potential_shaping(reach_now, self._prev_reach_progress)
        grasp_delta = potential_shaping(grasp_now, self._prev_grasp_progress)
        self._prev_reach_progress = reach_now
        self._prev_grasp_progress = grasp_now
        lift_potential = lift_progress(info["lift_height"], scale=scale, grasped=grasp_now)
        lift_prog = potential_shaping(lift_potential, self._prev_task_potential)
        self._prev_task_potential = lift_potential
        components = self.config.reward.compute_components(
            reach_progress=reach_delta,
            is_grasped=grasp_delta,
            task_progress=lift_prog,
            is_complete=info.get("success", False),
            action_delta_norm=info.get("action_delta_norm", 0.0),
            energy_norm=info.get("energy_norm", 0.0),
        )
        info["reward_components"] = components
        return sum(components.values())

    def _reach_to_target_reward(self, tcp_pos: np.ndarray, target_pos: np.ndarray) -> float:
        """Tanh-shaped reward for reaching a 3-D target position."""
        dist = float(np.linalg.norm(tcp_pos - target_pos))
        return reach_progress(dist, scale=self.config.reward.tanh_shaping_scale)

    def _task_reset(self) -> None:
        raise NotImplementedError

    def _get_obs(self) -> np.ndarray | dict[str, np.ndarray]:
        """Build observation from component list, optionally including camera images."""
        state = self._compute_obs_components()
        has_any_camera = self._wrist_renderer is not None or self._overhead_obs_renderer is not None
        if not has_any_camera:
            return state

        obs: dict[str, np.ndarray] = {}
        if self.config.obs_mode == "visual":
            self._privileged_state = state
            obs["state"] = self._get_current_qpos().astype(np.float32)
        else:
            obs["state"] = state

        if self._wrist_renderer is not None:
            self._wrist_renderer.update_scene(self.data, camera=self._wrist_cam_id)
            obs["wrist_camera"] = self._wrist_renderer.render()

        overhead_renderer = self._overhead_obs_renderer
        if overhead_renderer is not None:
            if self._overhead_obs_cam is None:
                raise RuntimeError("overhead camera id is not initialized")
            overhead_renderer.update_scene(self.data, camera=self._overhead_obs_cam)
            obs["overhead_camera"] = overhead_renderer.render()

        return obs

    def _get_info(self) -> dict:
        raise NotImplementedError

    def _compute_reward(self, info: dict) -> float:
        raise NotImplementedError
