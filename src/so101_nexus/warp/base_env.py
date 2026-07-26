"""Batched MuJoCo Warp base environment for SO101-Nexus tasks.

Natively batched: one model on the Warp device, one Data with a leading
``nworld`` (= ``num_envs``) dimension. State is read and written through
zero-copy ``wp.to_torch`` views, and observations/rewards/actions are torch
tensors on ``device``.

This extends the numpy-typed Gymnasium vector contract on two axes: (1)
obs/action/reward are torch tensors (no per-step host round-trip in the hot
path); and (2) autoreset is same-step (Brax/EnvPool style: the done step returns
post-reset obs), not Gymnasium 1.0's default ``AutoresetMode.NEXT_STEP``. The
``autoreset_mode`` metadata declares the latter so ``make_vec`` does not warn.

Backend divergence from the MuJoCo backend, in both cases measurable and not
configurable away. Task semantics, observation schema, and camera intrinsics ARE
in parity; these two are not.

**Physics** (see ``so101_nexus.scene``): mujoco_warp supports neither
``implicitfast`` nor ``noslip``, so the Warp scene uses the ``implicit``
integrator with no noslip. This is not a constant offset that a consumer can
calibrate out: it is contact-model-sensitive, so it shows up on tasks whose
success condition depends on sustained resting contact (pick-and-place, stack)
and not on tasks that do not (pick-lift). Measured downstream, the same
pick-and-place checkpoint loses 6-14 success points transferring Warp -> MuJoCo,
with the LARGER drop for smoother, gentler-contact policies. Validate any
"train in Warp, evaluate in MuJoCo" workflow on the MuJoCo backend before
trusting Warp-side success numbers.

**Rendering**: camera observations are NOT pixel-interchangeable with the MuJoCo
backend, even at bit-identical simulator state and camera pose. Both backends
build the same MJCF, and ``_setup_cameras`` corrects the two divergences that
are correctable from here (shadow casting, background colour), but mujoco_warp's
rasteriser ignores per-light ``diffuse`` and applies every active light at unit
intensity, so the Warp image is systematically brighter than MuJoCo's and clips
highlights. Do not train a vision policy on one backend and evaluate it on the
other without measuring the gap first;
``so101_nexus.testing.assert_render_parity`` exists to measure it.

Camera observations are supported through ``WristCamera`` and ``OverheadCamera``
components; Gymnasium ``render_mode`` is accepted only for compatibility and
ignored with a warning.
"""

from __future__ import annotations

import warnings
from typing import Any

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp
from gymnasium import spaces
from gymnasium.vector import AutoresetMode, VectorEnv
from gymnasium.vector.utils import batch_space

from so101_nexus.camera_utils import build_overhead_camera_mjcf
from so101_nexus.config import (
    EE_CONTROL_MODES,
    JOINT_CONTROL_MODES,
    SO101_ARM_JOINT_COUNT,
    SO101_JOINT_NAMES,
    SO101_TCP_SITE_NAME,
    ControlMode,
    EnvironmentConfig,
)
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
    GraspState,
    JointPositions,
    JointVelocities,
    OverheadCamera,
    WristCamera,
)
from so101_nexus.warp.render import unpack_rgb_uint8

# Normalized-delta physical scale (radians), shared with the MuJoCo backend's
# _DELTA_ACTION_SCALE: +/-0.05 for the five arm joints, +/-0.2 for the gripper.
_DELTA_ACTION_SCALE = (0.05, 0.05, 0.05, 0.05, 0.05, 0.2)

# Half-width of the pd_ee_pose position box, in metres. Sampling the reachable
# set gives a maximum TCP reach of 0.5457 m, so 0.55 m spans the workspace
# without admitting targets the solver could only clamp toward. Matches the
# MuJoCo backend's _EE_WORKSPACE_RADIUS.
_EE_WORKSPACE_RADIUS = 0.55

# Opaque black, packed ABGR as mujoco_warp's RenderContext.background_color
# expects. Matches the MuJoCo backend's clear colour for these skybox-free scenes.
_BACKGROUND_COLOR_ABGR = np.uint32(0xFF000000)


def _mat_to_quat(mat: torch.Tensor) -> torch.Tensor:
    """Batched rotation matrix ``(..., 3, 3)`` -> ``wxyz`` quaternion ``(..., 4)``.

    Shepperd's method, evaluated in the input dtype. Single precision is safe
    here because the branch is chosen so the divisor is always >= 2: there is no
    cancellation for extra precision to protect. Against ``mujoco.mju_mat2Quat``
    the float64 path agrees to 1e-9; on the float32 ``site_xmat`` views a float64
    interior moves the result by at most 1.5 float32 ulp while costing three
    float64 passes per inverse-kinematics iteration on hardware that runs them at
    1/64 rate. The result is canonicalized to ``w >= 0``.
    """
    m = mat
    m00, m11, m22 = m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]
    trace = m00 + m11 + m22

    def _safe(x: torch.Tensor) -> torch.Tensor:
        return torch.where(x.abs() < 1e-12, torch.ones_like(x), x)

    s0 = torch.sqrt(torch.clamp(trace, min=0.0) + 1.0) * 2.0
    q0 = torch.stack(
        [
            0.25 * s0,
            (m[..., 2, 1] - m[..., 1, 2]) / _safe(s0),
            (m[..., 0, 2] - m[..., 2, 0]) / _safe(s0),
            (m[..., 1, 0] - m[..., 0, 1]) / _safe(s0),
        ],
        dim=-1,
    )
    s1 = torch.sqrt(torch.clamp(1.0 + m00 - m11 - m22, min=0.0)) * 2.0
    q1 = torch.stack(
        [
            (m[..., 2, 1] - m[..., 1, 2]) / _safe(s1),
            0.25 * s1,
            (m[..., 0, 1] + m[..., 1, 0]) / _safe(s1),
            (m[..., 0, 2] + m[..., 2, 0]) / _safe(s1),
        ],
        dim=-1,
    )
    s2 = torch.sqrt(torch.clamp(1.0 - m00 + m11 - m22, min=0.0)) * 2.0
    q2 = torch.stack(
        [
            (m[..., 0, 2] - m[..., 2, 0]) / _safe(s2),
            (m[..., 0, 1] + m[..., 1, 0]) / _safe(s2),
            0.25 * s2,
            (m[..., 1, 2] + m[..., 2, 1]) / _safe(s2),
        ],
        dim=-1,
    )
    s3 = torch.sqrt(torch.clamp(1.0 - m00 - m11 + m22, min=0.0)) * 2.0
    q3 = torch.stack(
        [
            (m[..., 1, 0] - m[..., 0, 1]) / _safe(s3),
            (m[..., 0, 2] + m[..., 2, 0]) / _safe(s3),
            (m[..., 1, 2] + m[..., 2, 1]) / _safe(s3),
            0.25 * s3,
        ],
        dim=-1,
    )
    cond0 = trace > 0.0
    cond1 = (m00 >= m11) & (m00 >= m22)
    cond2 = m11 >= m22
    quat = torch.where(
        cond0[..., None],
        q0,
        torch.where(cond1[..., None], q1, torch.where(cond2[..., None], q2, q3)),
    )
    quat = quat / _safe(torch.linalg.norm(quat, dim=-1, keepdim=True))
    return torch.where(quat[..., 0:1] < 0.0, -quat, quat)


def _grasp_from_contacts(
    *,
    contact_geom: torch.Tensor,
    contact_world: torch.Tensor,
    normal_force: torch.Tensor,
    nacon: int,
    obj_geom: torch.Tensor,
    gripper_mask: torch.Tensor,
    jaw_mask: torch.Tensor,
    threshold: float,
    num_envs: int,
) -> torch.Tensor:
    """Reduce flat contacts to a ``(num_envs,)`` two-sided grasp signal in {0, 1}.

    A world grasps when its target geom (``obj_geom[world]``) contacts both a
    gripper finger geom and a moving-jaw finger geom, each with normal force at or
    above ``threshold``. Pure tensor reduction over the packed ``[0, nacon)``
    contact slots, so it is unit-testable with synthetic arrays. Mirrors the
    MuJoCo base's ``_is_grasping``.
    """
    if nacon == 0:
        return torch.zeros(num_envs, device=obj_geom.device)
    geom = contact_geom[:nacon].long()
    world = contact_world[:nacon].long().clamp(0, num_envs - 1)
    g1, g2 = geom[:, 0], geom[:, 1]
    obj = obj_geom[world]
    obj_is_g1 = g1 == obj
    involved = obj_is_g1 | (g2 == obj)
    other = torch.where(obj_is_g1, g2, g1).clamp(min=0)
    strong = involved & (normal_force[:nacon] >= threshold)
    grip_hit = (gripper_mask[other] & strong).to(torch.float32)
    jaw_hit = (jaw_mask[other] & strong).to(torch.float32)
    grip_w = torch.zeros(num_envs, device=obj_geom.device)
    jaw_w = torch.zeros(num_envs, device=obj_geom.device)
    grip_w.scatter_reduce_(0, world, grip_hit, reduce="amax")
    jaw_w.scatter_reduce_(0, world, jaw_hit, reduce="amax")
    return (grip_w.bool() & jaw_w.bool()).to(torch.float32)


class SO101NexusWarpVectorEnv(VectorEnv):
    """Shared GPU-batched Warp base class for SO101-Nexus tasks."""

    metadata = {"render_modes": [], "autoreset_mode": AutoresetMode.SAME_STEP}
    _N_SUBSTEPS = 4
    _VALID_CONTROL_MODES = frozenset(JOINT_CONTROL_MODES + EE_CONTROL_MODES)

    def __init__(  # noqa: PLR0915
        self,
        *,
        num_envs: int,
        config: EnvironmentConfig,
        mjm: mujoco.MjModel,
        control_mode: ControlMode = "pd_joint_pos",
        device: str = "cuda",
        max_episode_steps: int = 512,
        seed: int | None = None,
        nconmax: int | None = None,
        njmax: int | None = None,
        render_mode: str | None = None,
    ) -> None:
        if control_mode not in self._VALID_CONTROL_MODES:
            raise ValueError(
                f"control_mode must be one of {sorted(self._VALID_CONTROL_MODES)}, "
                f"got {control_mode!r}"
            )
        if render_mode is not None:
            warnings.warn(
                "render_mode is ignored by the Warp backend because Warp vector envs "
                "do not implement render(); configure WristCamera or OverheadCamera "
                "observations for image tensors instead.",
                UserWarning,
                stacklevel=2,
            )
        if config.observations is not None:
            self._validate_obs_components(config.observations)

        self.config = config
        self.control_mode = control_mode
        self.render_mode = None
        self.max_episode_steps = max_episode_steps
        self.robot_init_qpos_noise = config.robot_init_qpos_noise
        self.device = torch.device(device)
        self._wp_device = wp.get_device(
            "cpu" if self.device.type == "cpu" else f"cuda:{self.device.index or 0}"
        )

        self.mjm = mjm
        mjd = mujoco.MjData(mjm)
        mujoco.mj_forward(mjm, mjd)
        with wp.ScopedDevice(self._wp_device):
            self.model = mjw.put_model(mjm)
            # nconmax/njmax default to None (mujoco_warp auto-sizes per world);
            # tasks should pass generous explicit budgets (auto-size is too small
            # under active control, which causes nefc overflow and physics drop).
            self.data = mjw.put_data(mjm, mjd, nworld=num_envs, nconmax=nconmax, njmax=njmax)

        # Zero-copy torch views. NEVER rebind these; mutate in place only.
        self.qpos = wp.to_torch(self.data.qpos)  # (N, nq)
        self.qvel = wp.to_torch(self.data.qvel)  # (N, nv)
        self.ctrl = wp.to_torch(self.data.ctrl)  # (N, nu)
        self.site_xpos = wp.to_torch(self.data.site_xpos)  # (N, nsite, 3)
        self.site_xmat = wp.to_torch(self.data.site_xmat)  # (N, nsite, 3, 3)

        joint_ids = [
            mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, n) for n in SO101_JOINT_NAMES
        ]
        act_ids = [
            mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in SO101_JOINT_NAMES
        ]
        self._qpos_adr = torch.as_tensor(
            [mjm.jnt_qposadr[j] for j in joint_ids], device=self.device
        )
        self._dof_adr = torch.as_tensor([mjm.jnt_dofadr[j] for j in joint_ids], device=self.device)
        self._act_ids = torch.as_tensor(act_ids, device=self.device)
        self._tcp_site_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_SITE, SO101_TCP_SITE_NAME)

        ctrl_range = mjm.actuator_ctrlrange[np.asarray(act_ids)]
        jnt_range = mjm.jnt_range[np.asarray(joint_ids)]
        low = np.maximum(ctrl_range[:, 0], jnt_range[:, 0]).astype(np.float32)
        high = np.minimum(ctrl_range[:, 1], jnt_range[:, 1]).astype(np.float32)
        self._target_low = torch.as_tensor(low, device=self.device)
        self._target_high = torch.as_tensor(high, device=self.device)
        self._rest_qpos = torch.as_tensor(
            np.asarray(config.robot.rest_qpos_rad, dtype=np.float32), device=self.device
        )
        self._delta_scale = torch.as_tensor(
            np.asarray(_DELTA_ACTION_SCALE, dtype=np.float32), device=self.device
        )

        n_joints = len(SO101_JOINT_NAMES)
        if control_mode == "pd_joint_pos":
            self.single_action_space = spaces.Box(low=low, high=high, dtype=np.float32)
            self._action_low = self._target_low
            self._action_high = self._target_high
        elif control_mode == "pd_ee_pose":
            # Absolute TCP pose: world position, orientation as a rotation
            # vector, then the gripper joint target on pd_joint_pos's bounds.
            ee_low = np.concatenate(
                [np.full(3, -_EE_WORKSPACE_RADIUS), np.full(3, -np.pi), low[-1:]]
            ).astype(np.float32)
            ee_high = np.concatenate(
                [np.full(3, _EE_WORKSPACE_RADIUS), np.full(3, np.pi), high[-1:]]
            ).astype(np.float32)
            self.single_action_space = spaces.Box(low=ee_low, high=ee_high, dtype=np.float32)
            self._action_low = torch.as_tensor(ee_low, device=self.device)
            self._action_high = torch.as_tensor(ee_high, device=self.device)
        else:
            # Delta modes expose a normalized [-1, 1] box; the physical scale is
            # _DELTA_ACTION_SCALE (joints) or EE_DELTA_ACTION_SCALE (end effector).
            n_actions = EE_ACTION_DIM if control_mode == "pd_ee_delta_pose" else n_joints
            self.single_action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(n_actions,), dtype=np.float32
            )
            self._action_low = torch.full((n_actions,), -1.0, device=self.device)
            self._action_high = torch.full((n_actions,), 1.0, device=self.device)
        self.single_observation_space = spaces.Box(
            -np.inf, np.inf, shape=(self._obs_dim(),), dtype=np.float32
        )
        self.num_envs = num_envs
        self.action_space = batch_space(self.single_action_space, num_envs)
        self.observation_space = batch_space(self.single_observation_space, num_envs)
        # Per-world task descriptions (decision 8): heterogeneous worlds may carry
        # different tasks. Subclasses populate this; the scalar ``task_description``
        # property reduces it. Default empty until a subclass sets descriptions.
        self.task_descriptions: list[str] = [""] * num_envs

        self._generator = torch.Generator(device=self.device)
        if seed is not None:
            self._generator.manual_seed(seed)
        self._elapsed = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._prev_action: torch.Tensor | None = None
        self._has_prev_action = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._prev_target = self._rest_qpos.expand(num_envs, n_joints).clone()

        # Arm DOF addresses (first five joints; the gripper is excluded), mirroring
        # the MuJoCo base's _arm_qvel_addrs for the static-robot check.
        self._arm_dof_adr = self._dof_adr[:-1]
        # Finger contact geoms for grasp detection (condim==6 surfaces on the
        # gripper and moving-jaw bodies) as boolean per-geom masks.
        ngeom = mjm.ngeom
        gripper_bid = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "gripper")
        jaw_bid = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "moving_jaw_so101_v1")
        self._gripper_mask = self._finger_geom_mask(mjm, gripper_bid, ngeom)
        self._jaw_mask = self._finger_geom_mask(mjm, jaw_bid, ngeom)
        # Per-world target geom for grasp detection; manipulation tasks set this to
        # a (num_envs,) long tensor. None means no graspable object (primitives).
        self._obj_geom: torch.Tensor | None = None
        # Zero-copy contact views; the force buffer is allocated lazily on first
        # grasp query so primitive envs pay nothing.
        self._contact_geom_view = wp.to_torch(self.data.contact.geom)  # (naconmax, 2)
        self._contact_world_view = wp.to_torch(self.data.contact.worldid)  # (naconmax,)
        self._nacon_view = wp.to_torch(self.data.nacon)  # (1,)
        self._contact_ids: wp.array | None = None
        self._force_buf: wp.array | None = None
        self._force_view: torch.Tensor | None = None
        self._ik_graph: torch.cuda.CUDAGraph | None = None
        if control_mode in EE_CONTROL_MODES:
            self._setup_ee_control()
        self._setup_cameras()
        self._step_graph = None
        self._capture_step_graph()
        self._capture_ik_graph()

    def _setup_ee_control(self) -> None:
        """Allocate the persistent batched inverse-kinematics state for the EE modes.

        The Jacobian, tool-point, and body-id arrays are sized once here because
        ``mujoco_warp.jac`` writes into caller-owned arrays: allocating them per
        step would charge every control step for ``num_envs``-sized allocations.
        The command and answer buffers are persistent for the same reason and,
        more importantly, because the CUDA graph captured over the solve binds
        their addresses once (see ``_capture_ik_graph``).
        """
        robot = self.config.robot
        self._ee_delta_scale = torch.as_tensor(
            np.asarray(robot.ee_delta_action_scale, dtype=np.float32), device=self.device
        )
        self._ee_orientation_weight = robot.ee_orientation_weight
        self._arm_qpos_adr = self._qpos_adr[:-1]
        nv = self.mjm.nv
        with wp.ScopedDevice(self._wp_device):
            # mujoco_warp.jac writes (nworld, 3, nv); a (nworld, nv, 3) buffer is
            # accepted but silently filled as if transposed.
            self._ik_jacp = wp.zeros((self.num_envs, 3, nv), dtype=wp.float32)
            self._ik_jacr = wp.zeros((self.num_envs, 3, nv), dtype=wp.float32)
            self._ik_point = wp.zeros(self.num_envs, dtype=wp.vec3f)
            self._ik_body = wp.array(
                np.full(self.num_envs, self.mjm.site_bodyid[self._tcp_site_id], dtype=np.int32),
                dtype=wp.int32,
            )
        self._ik_jacp_view = wp.to_torch(self._ik_jacp)  # (N, 3, nv)
        self._ik_jacr_view = wp.to_torch(self._ik_jacr)  # (N, 3, nv)
        self._ik_point_view = wp.to_torch(self._ik_point)  # (N, 3)
        # Graph-stable solve I/O: the command goes in, the arm targets come out.
        self._ik_target_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self._ik_target_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self._ik_arm_target = torch.zeros(
            (self.num_envs, SO101_ARM_JOINT_COUNT), device=self.device
        )

    def _setup_cameras(self) -> None:
        """Detect camera components and configure batched rendering (no-op if none).

        State-only configs pay nothing. When a ``WristCamera``/``OverheadCamera``
        component is present, builds a mujoco_warp ``RenderContext``, reallocates
        per-world camera model arrays for wrist domain randomization, and rebuilds
        the observation space as a ``Dict`` matching the MuJoCo backend. Runs
        before CUDA-graph capture so the captured step binds the per-world arrays.

        The observation SCHEMA and the camera intrinsics/extrinsics match the
        MuJoCo backend; the rendered pixels do not. See this module's docstring.
        """
        obs = self.config.observations or []
        self._wrist_cam: WristCamera | None = next(
            (c for c in obs if isinstance(c, WristCamera)), None
        )
        self._overhead_cam: OverheadCamera | None = next(
            (c for c in obs if isinstance(c, OverheadCamera)), None
        )
        self._has_cameras = self._wrist_cam is not None or self._overhead_cam is not None
        self._image_bufs: list = []
        self._privileged_state: torch.Tensor | None = None
        if not self._has_cameras:
            return

        mjm = self.mjm
        # (component, mujoco camera id) in declaration order (wrist, then overhead).
        self._cam_specs: list[tuple[CameraObservation, int]] = []
        if self._wrist_cam is not None:
            wid = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
            if wid < 0:
                raise RuntimeError("WristCamera requested but 'wrist_cam' is not in the model")
            self._cam_specs.append((self._wrist_cam, wid))
            self._wrist_mjid = wid
        if self._overhead_cam is not None:
            oid = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_CAMERA, "overhead_cam")
            if oid < 0:
                raise RuntimeError(
                    "OverheadCamera requested but 'overhead_cam' is not in the scene; "
                    "the task must inject it via camera_utils.build_overhead_camera_mjcf"
                )
            self._cam_specs.append((self._overhead_cam, oid))

        # Active render index = position within the ascending active-id list, the
        # ordering mujoco_warp.create_render_context assigns from cam_active.
        active_ids = sorted(cid for _, cid in self._cam_specs)
        cam_active = [i in active_ids for i in range(mjm.ncam)]
        spec_by_id = {cid: comp for comp, cid in self._cam_specs}
        cam_res = [(spec_by_id[cid].width, spec_by_id[cid].height) for cid in active_ids]
        self._render_index = {cid: active_ids.index(cid) for cid in active_ids}
        # Per-world fovy randomization (wrist DR) requires rays recomputed per world.
        use_precomputed_rays = self._wrist_cam is None
        with wp.ScopedDevice(self._wp_device):
            # use_shadows defaults False, which silently drops the shadows the
            # scene's key light asks for via castshadow (see SCENE_LIGHTS_XML) and
            # is a first-order mismatch against the MuJoCo backend's image: the
            # cast shadow under the gripper is plainly visible there and nearly
            # absent without this. Measured cost is a few percent of step time.
            self._render_ctx = mjw.create_render_context(
                mjm,
                nworld=self.num_envs,
                cam_res=cam_res,
                render_rgb=True,
                render_depth=False,
                render_seg=False,
                cam_active=cam_active,
                use_precomputed_rays=use_precomputed_rays,
                use_shadows=True,
            )
            # mujoco_warp clears to a hardcoded blue-tinted (0.1, 0.1, 0.2) that
            # corresponds to nothing in the model; these scenes carry no skybox,
            # so MuJoCo clears to black. Packed ABGR uint32, matching io.py's
            # pack_rgba_to_uint32 at render-context construction.
            self._render_ctx.background_color = _BACKGROUND_COLOR_ABGR
            if self._wrist_cam is not None:
                self._reallocate_per_world_cameras(mjm)
        self._build_camera_observation_space()

    def _reallocate_per_world_cameras(self, mjm: mujoco.MjModel) -> None:
        """Give the model per-world camera pose/fovy arrays for wrist DR.

        ``put_model`` shares these across worlds (leading dim 1); reallocating to
        ``num_envs`` rows lets ``camlight`` and the renderer read per-world values
        (kernels index ``worldid % shape[0]``). Quaternions stay in MuJoCo ``wxyz``
        order (scalar first), matching ``put_model``.
        """
        ncam, n = mjm.ncam, self.num_envs
        cam_pos = np.broadcast_to(mjm.cam_pos, (n, ncam, 3)).copy()
        cam_quat = np.broadcast_to(mjm.cam_quat, (n, ncam, 4)).copy()
        cam_fovy = np.broadcast_to(mjm.cam_fovy, (n, ncam)).copy()
        self.model.cam_pos = wp.array(cam_pos, dtype=wp.vec3)
        self.model.cam_quat = wp.array(cam_quat, dtype=wp.quat)
        self.model.cam_fovy = wp.array(cam_fovy, dtype=wp.float32)
        self._cam_pos = wp.to_torch(self.model.cam_pos)  # (N, ncam, 3)
        self._cam_quat = wp.to_torch(self.model.cam_quat)  # (N, ncam, 4) wxyz
        self._cam_fovy = wp.to_torch(self.model.cam_fovy)  # (N, ncam)

    def _build_camera_observation_space(self) -> None:
        state_size = len(SO101_JOINT_NAMES) if self.config.obs_mode == "visual" else self._obs_dim()
        obs_spaces: dict[str, spaces.Space] = {
            "state": spaces.Box(-np.inf, np.inf, shape=(state_size,), dtype=np.float32),
        }
        for comp, _ in self._cam_specs:
            obs_spaces[comp.name] = spaces.Box(
                low=0, high=255, shape=(comp.height, comp.width, 3), dtype=np.uint8
            )
        self.single_observation_space = spaces.Dict(obs_spaces)
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)

    def _render_camera_images(self) -> dict[str, torch.Tensor]:
        """Render all active cameras and return ``name -> (N, H, W, 3)`` uint8 tensors."""
        self._image_bufs = []
        images: dict[str, torch.Tensor] = {}
        with wp.ScopedDevice(self._wp_device):
            self._update_render_markers()
            mjw.refit_bvh(self.model, self.data, self._render_ctx)
            mjw.render(self.model, self.data, self._render_ctx)
            for comp, cid in self._cam_specs:
                buf = wp.empty((self.num_envs, comp.height, comp.width, 3), dtype=wp.uint8)
                unpack_rgb_uint8(self._render_ctx, self._render_index[cid], buf)
                self._image_bufs.append(buf)  # keep alive for the returned torch views
                images[comp.name] = wp.to_torch(buf)
        return images

    def _update_render_markers(self) -> None:
        """Refresh per-world visual-only marker geom poses before rendering (default no-op).

        Default no-op. Tasks whose target has no physical body (LookAt, Move)
        override this to write the marker geom's ``geom_xpos`` so the camera image
        shows the goal, matching the MuJoCo backend. Runs after the step's physics
        forward and immediately before ``refit_bvh``/``render``, so the override is
        not clobbered by a later kinematics pass.
        """

    def _randomize_wrist_camera(self, idx: torch.Tensor) -> None:
        """Per-world wrist camera DR (pitch, position, fovy) for the reset indices.

        Mirrors ``SO101NexusMuJoCoBaseEnv._randomize_wrist_camera`` with the seeded
        generator. Writes the per-world model arrays in place (``wxyz`` quaternions),
        so the next ``camlight``/render reflects them.
        """
        n = int(idx.numel())
        if n == 0 or self._wrist_cam is None:
            return
        wc = self._wrist_cam
        cid = self._wrist_mjid
        g = self._generator
        pitch_lo, pitch_hi = wc.pitch_rad_range
        pitch = torch.rand(n, generator=g, device=self.device) * (pitch_hi - pitch_lo) + pitch_lo
        half = pitch * 0.5
        quat = torch.zeros((n, 4), device=self.device)
        quat[:, 0] = torch.cos(half)  # w
        quat[:, 1] = torch.sin(half)  # x (rotation about camera X = pitch)
        self._cam_quat[idx, cid] = quat
        u = torch.rand((n, 3), generator=g, device=self.device) * 2.0 - 1.0
        pos = torch.empty((n, 3), device=self.device)
        pos[:, 0] = u[:, 0] * wc.pos_x_noise
        pos[:, 1] = wc.pos_y_center + u[:, 1] * wc.pos_y_noise
        pos[:, 2] = wc.pos_z_center + u[:, 2] * wc.pos_z_noise
        self._cam_pos[idx, cid] = pos
        fov_lo, fov_hi = wc.fov_deg_range
        self._cam_fovy[idx, cid] = (
            torch.rand(n, generator=g, device=self.device) * (fov_hi - fov_lo) + fov_lo
        )

    @staticmethod
    def _overhead_camera_xml(config: EnvironmentConfig) -> str:
        """Return the overhead ``<camera>`` MJCF for the scene, or '' if not requested.

        Subclasses call this before ``super().__init__`` to inject a world-fixed
        overhead camera into the scene worldbody when an ``OverheadCamera``
        observation is configured (the Warp renderer rasterizes model cameras
        only). Framing reuses ``camera_utils`` so both backends frame the scene
        identically; the shading of what they frame still differs (see the module
        docstring).
        """
        cam = next(
            (c for c in (config.observations or []) if isinstance(c, OverheadCamera)),
            None,
        )
        if cam is None:
            return ""
        return build_overhead_camera_mjcf(
            spawn_center=config.spawn_center,
            spawn_max_radius=config.spawn_max_radius,
            fov_deg=cam.fov_deg,
            width=cam.width,
            height=cam.height,
        )

    def _capture_step_graph(self) -> None:
        """Capture the per-step substep loop into a CUDA graph for replay.

        ``mujoco_warp.step`` is a collection of small kernel launches; on CUDA,
        replaying a captured graph removes per-launch overhead (the throughput
        win Warp exists for). The graph references the persistent ``data``
        buffers, so the in-place ``ctrl`` write before each replay is honored.
        CPU has no graph support, so stepping falls back to the direct loop. The
        warmup/capture advances physics from the construction state, which the
        first ``reset()`` overwrites before any episode begins.
        """
        if self.device.type != "cuda":
            return
        try:
            with wp.ScopedDevice(self._wp_device):
                for _ in range(self._N_SUBSTEPS):
                    mjw.step(self.model, self.data)
                with wp.ScopedCapture() as capture:
                    for _ in range(self._N_SUBSTEPS):
                        mjw.step(self.model, self.data)
            self._step_graph = capture.graph
        except Exception as exc:  # capture is an optimization; never block construction
            warnings.warn(
                f"CUDA graph capture failed ({exc}); using direct stepping.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._step_graph = None

    def _capture_ik_graph(self) -> None:
        """Capture the fixed-iteration inverse-kinematics solve into a CUDA graph.

        ``_solve_ee_ik`` is three iterations of small Warp kernels interleaved
        with small torch reductions, so like ``mujoco_warp.step`` it is
        launch-bound rather than compute-bound: replaying it as one graph removes
        the per-launch overhead that dominates it.

        Capture is driven by ``torch.cuda.graph`` rather than
        ``wp.ScopedCapture`` because the loop allocates torch intermediates. Only
        a capture torch itself opened routes those allocations to the graph's
        private memory pool; under a Warp-opened capture the caching allocator
        would reach for ``cudaMalloc`` and abort the capture. Warp launches are
        steered onto the capturing stream with ``ScopedStream``, whose entry and
        exit synchronization is disabled because cross-stream event waits are
        illegal mid-capture and there is nothing outstanding to order against:
        ``torch.cuda.graph`` synchronizes the device on entry.

        Replay runs on torch's default stream, which Warp's device stream
        implicitly synchronizes with, so it stays ordered against
        ``_advance_physics`` exactly as the eager solve was.

        CPU has no graph support, and capture failure is never fatal: both fall
        back to calling ``_solve_ee_ik`` directly.
        """
        if self.control_mode not in EE_CONTROL_MODES or self.device.type != "cuda":
            return
        try:
            # Warm up on the same stream the eager path uses, so lazy module
            # loads and linear-algebra workspace allocations are done before the
            # capture opens. Identity orientation keeps the warmup solve well
            # posed; the values are irrelevant, only the work they trigger.
            self._ik_target_pos.copy_(self._tcp_pos())
            self._ik_target_quat.zero_()
            self._ik_target_quat[:, 0] = 1.0
            with wp.ScopedDevice(self._wp_device):
                self._solve_ee_ik()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, capture_error_mode="thread_local"):
                capture_stream = wp.stream_from_torch(torch.cuda.current_stream())
                with (
                    wp.ScopedDevice(self._wp_device),
                    wp.ScopedStream(capture_stream, sync_enter=False, sync_exit=False),
                ):
                    self._solve_ee_ik()
            self._ik_graph = graph
        except Exception as exc:  # capture is an optimization; never block construction
            warnings.warn(
                f"Inverse-kinematics CUDA graph capture failed ({exc}); using the direct solve.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._ik_graph = None

    def _advance_physics(self) -> None:
        """Advance ``_N_SUBSTEPS`` of physics via the captured graph or direct loop."""
        if self._step_graph is not None:
            wp.capture_launch(self._step_graph)
        else:
            for _ in range(self._N_SUBSTEPS):
                mjw.step(self.model, self.data)

    def _validate_obs_components(self, observations) -> None:
        """Reject unsupported observation components at construction (fail fast).

        The Warp base routes the robot-generic components (``JointPositions``,
        ``JointVelocities``, ``EndEffectorPose``, ``GraspState``) and camera
        components (``WristCamera``, ``OverheadCamera``) centrally, mirroring the
        MuJoCo base, and delegates task-specific components to
        ``_get_component_data`` (a subclass declares those via
        ``_supported_obs_components``). Anything else raises here rather than at
        the first reset, so the error names the unsupported component upfront.
        """
        supported = {
            JointPositions,
            JointVelocities,
            EndEffectorPose,
            GraspState,
            WristCamera,
            OverheadCamera,
            *self._supported_obs_components(),
        }
        for comp in observations:
            if not isinstance(comp, tuple(supported)):
                raise NotImplementedError(
                    f"{type(self).__name__} does not support observation "
                    f"component {comp!r} on the Warp backend"
                )

    @property
    def control_dt(self) -> float:
        """Simulated seconds advanced by one ``step()`` (physics timestep x substeps).

        This is the time between consecutive observations, and therefore the
        correct ``dt`` for finite-differencing recorded joint positions. It is
        unrelated to a teleop recording's wall-clock fps.
        """
        return float(self.mjm.opt.timestep) * self._N_SUBSTEPS

    def _obs_dim(self) -> int:
        if self.config.observations is None:
            raise RuntimeError("config.observations must be set")
        return sum(c.size for c in self.config.observations if c.size > 0)

    def _joint_qpos(self) -> torch.Tensor:
        return self.qpos.index_select(1, self._qpos_adr)

    def _joint_qvel(self) -> torch.Tensor:
        return self.qvel.index_select(1, self._dof_adr)

    def _tcp_pos(self) -> torch.Tensor:
        return self.site_xpos[:, self._tcp_site_id, :]

    def _compute_state_vector(self) -> torch.Tensor:
        """Concatenate the flat state components (camera components are skipped)."""
        if self.config.observations is None:
            raise RuntimeError("config.observations must be set")
        parts: list[torch.Tensor] = []
        for comp in self.config.observations:
            if isinstance(comp, JointPositions):
                parts.append(self._joint_qpos())
            elif isinstance(comp, JointVelocities):
                parts.append(self._joint_qvel())
            elif isinstance(comp, EndEffectorPose):
                parts.append(self._get_tcp_pose7())
            elif isinstance(comp, GraspState):
                parts.append(self._is_grasping().unsqueeze(1))
            elif isinstance(comp, CameraObservation):
                continue
            else:
                parts.append(self._get_component_data(comp))
        if not parts:  # camera-only config: empty flat state
            return torch.zeros((self.num_envs, 0), device=self.device)
        return torch.cat(parts, dim=1).to(torch.float32)

    def _compute_obs(self) -> torch.Tensor | dict[str, torch.Tensor]:
        """Return flat state, or a dict obs with batched images when cameras are active."""
        state = self._compute_state_vector()
        if not self._has_cameras:
            return state
        obs: dict[str, torch.Tensor] = {}
        if self.config.obs_mode == "visual":
            self._privileged_state = state
            obs["state"] = self._joint_qpos()
        else:
            obs["state"] = state
        obs.update(self._render_camera_images())
        return obs

    def _write_reset_state(self, mask: torch.Tensor, init_qpos: torch.Tensor | None = None) -> None:
        idx = mask.nonzero(as_tuple=True)[0]
        n = int(idx.numel())
        if n == 0:
            return
        target = self._sample_reset_qpos(n, init_qpos)
        rows = idx[:, None]
        self.qpos[rows, self._qpos_adr] = target
        self.qvel[rows, self._dof_adr] = 0.0
        self.ctrl[rows, self._act_ids] = target
        self._prev_target[idx] = target
        self._elapsed[idx] = 0
        self._task_reset(mask)
        if self._wrist_cam is not None:
            self._randomize_wrist_camera(idx)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[torch.Tensor | dict[str, torch.Tensor], dict]:
        """Reset all worlds and return the initial batched observation and info."""
        if seed is not None:
            self._generator.manual_seed(seed)
        init_qpos = self._parse_init_qpos(options)
        self._prev_action = None
        self._has_prev_action.fill_(False)
        mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        with wp.ScopedDevice(self._wp_device):
            self._write_reset_state(mask, init_qpos=init_qpos)
            mjw.forward(self.model, self.data)
            # Capture the reset reference BEFORE settling so it is settle-independent
            # and identical to the same-step autoreset path (which cannot settle a
            # subset of worlds). Settling only warms up the robot for the returned
            # observation; it must not move task reference state (targets, baselines).
            self._refresh_reset_reference_state(mask)
            for _ in range(self.config.reset_settle_frames):
                for _ in range(self._N_SUBSTEPS):
                    mjw.step(self.model, self.data)
        obs = self._compute_obs()
        info: dict[str, Any] = {"task_description": tuple(self.task_descriptions)}
        if self._privileged_state is not None:
            info["privileged_state"] = self._privileged_state
        return obs, info

    def close(self, **kwargs: Any) -> None:
        """No-op: Warp device memory is released when the env is garbage-collected."""

    @property
    def task_description(self) -> str:
        """Scalar task description: the shared string when worlds agree, else generic."""
        descs = self.task_descriptions
        if not descs:
            return ""
        first = descs[0]
        if all(d == first for d in descs):
            return first
        return self._generic_task_description()

    def _generic_task_description(self) -> str:
        """Family-level fallback when worlds carry heterogeneous task strings."""
        return "Complete the task."

    def _action_to_ctrl(self, action: torch.Tensor) -> torch.Tensor:
        if self.control_mode == "pd_joint_pos":
            return torch.clamp(action, self._target_low, self._target_high)
        if self.control_mode in EE_CONTROL_MODES:
            return self._ee_action_to_ctrl(action)
        delta = action * self._delta_scale
        if self.control_mode == "pd_joint_delta_pos":
            return torch.clamp(self._joint_qpos() + delta, self._target_low, self._target_high)
        self._prev_target = torch.clamp(
            self._prev_target + delta, self._target_low, self._target_high
        )
        return self._prev_target

    def _ee_action_to_ctrl(self, action: torch.Tensor) -> torch.Tensor:
        """Resolve an ``(N, 7)`` end-effector action to ``(N, 6)`` joint targets.

        ``pd_ee_pose`` reads the action as an absolute world TCP pose (position
        plus rotation vector); ``pd_ee_delta_pose`` scales the normalized action
        by ``config.robot.ee_delta_action_scale`` and applies it to the
        *measured* pose, mirroring ``pd_joint_delta_pos`` rather than the held
        target. The gripper rides along as a plain joint target in both cases.
        """
        pose = self._get_tcp_pose7()
        if self.control_mode == "pd_ee_pose":
            target_pos = action[:, :3]
            target_quat = rotvec_to_quat(action[:, 3:6])
            gripper = action[:, 6]
        else:
            scaled = action * self._ee_delta_scale
            target_pos = pose[:, :3] + scaled[:, :3]
            # Left-multiply: the rotation delta composes about the world axes, so
            # it reads in the same frame as the position delta beside it.
            target_quat = quat_multiply(rotvec_to_quat(scaled[:, 3:6]), pose[:, 3:])
            gripper = self._joint_qpos()[:, -1] + scaled[:, 6]
        arm_target = self._ee_ik_arm_targets(target_pos, target_quat)
        gripper = torch.clamp(gripper, self._target_low[-1], self._target_high[-1])
        return torch.cat([arm_target, gripper.unsqueeze(1)], dim=1)

    def _ee_ik_arm_targets(
        self, target_pos: torch.Tensor, target_quat: torch.Tensor
    ) -> torch.Tensor:
        """Batched damped-least-squares IK: ``(N, 5)`` arm joint targets.

        Copies the command into the persistent solve buffers, then either
        replays the captured CUDA graph or runs ``_solve_ee_ik`` directly. The
        returned tensor is ``_ik_arm_target`` itself, which the next solve
        overwrites, so callers must consume it before the next control step.
        """
        self._ik_target_pos.copy_(target_pos)
        self._ik_target_quat.copy_(target_quat)
        if self._ik_graph is not None:
            self._ik_graph.replay()
        else:
            with wp.ScopedDevice(self._wp_device):
                self._solve_ee_ik()
        return self._ik_arm_target

    def _solve_ee_ik(self) -> None:
        """Solve ``_ik_target_pos``/``_ik_target_quat`` into ``_ik_arm_target``.

        Warm-starts from the current arm configuration and takes
        ``EE_IK_ITERATIONS`` steps, re-evaluating forward kinematics and the tool
        Jacobian at the working configuration each time. The arm's tool Jacobian
        is rank 5, so orientation is de-weighted rather than tracked exactly (see
        ``config.robot.ee_orientation_weight``), and targets outside the
        reachable set resolve to the closest pose the joint limits allow.

        Shape-static and fixed-iteration by construction, and it reads and writes
        only persistent buffers, which is what makes ``_capture_ik_graph`` able to
        record it. Any allocation, host synchronization, or data-dependent
        iteration count added here breaks that capture.

        ``qpos`` is restored on exit, but the derived kinematics (``site_xpos``,
        ``site_xmat``, ``subtree_com``) are left at the last working
        configuration. ``step`` calls this immediately before ``_advance_physics``,
        whose ``mjw.step`` recomputes them, so nothing in the env observes the
        difference; an out-of-band caller must run ``mjw.forward`` first to get the
        measured pose as the delta reference.
        """
        # index_select copies, and arm_q is rebound (never mutated in place)
        # below, so start_q holds the entry configuration for free.
        start_q = self.qpos.index_select(1, self._arm_qpos_adr)
        arm_q = start_q
        low, high = self._target_low[:-1], self._target_high[:-1]
        for _ in range(EE_IK_ITERATIONS):
            self.qpos[:, self._arm_qpos_adr] = arm_q
            # com_pos is required as well: mjw.jac reads subtree_com, and
            # kinematics alone leaves it stale (rotational rows come out wrong).
            mjw.kinematics(self.model, self.data)
            mjw.com_pos(self.model, self.data)
            pose = self._get_tcp_pose7()
            self._ik_point_view.copy_(pose[:, :3])
            mjw.jac(
                self.model,
                self.data,
                self._ik_jacp,
                self._ik_jacr,
                self._ik_point,
                self._ik_body,
            )
            jac = torch.cat([self._ik_jacp_view, self._ik_jacr_view], dim=1).index_select(
                2, self._arm_dof_adr
            )
            delta_q = ee_ik_delta_q(
                jac,
                pose[:, :3],
                pose[:, 3:],
                self._ik_target_pos,
                self._ik_target_quat,
                orientation_weight=self._ee_orientation_weight,
            )
            arm_q = torch.clamp(arm_q + delta_q, low, high)
        # The loop writes only the arm block, and mjw.step in _advance_physics
        # recomputes every derived field, so restoring qpos restores the state.
        self.qpos[:, self._arm_qpos_adr] = start_q
        self._ik_arm_target.copy_(arm_q)

    def step(
        self, actions: torch.Tensor
    ) -> tuple[
        torch.Tensor | dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, dict
    ]:
        """Apply actions to all worlds, advance physics, autoreset done worlds."""
        public_action = torch.as_tensor(actions, device=self.device, dtype=torch.float32)
        energy_norm = torch.linalg.norm(public_action, dim=1)
        if self._prev_action is None:
            action_delta_norm = torch.zeros(self.num_envs, device=self.device)
        else:
            raw_delta = torch.linalg.norm(public_action - self._prev_action, dim=1)
            action_delta_norm = torch.where(
                self._has_prev_action, raw_delta, torch.zeros_like(raw_delta)
            )
        self._prev_action = public_action.clone()
        self._has_prev_action.fill_(True)

        clipped = torch.clamp(public_action, self._action_low, self._action_high)
        self.ctrl[:, self._act_ids] = self._action_to_ctrl(clipped)
        with wp.ScopedDevice(self._wp_device):
            self._advance_physics()
        self._elapsed += 1

        reward, success, info = self._compute_reward_terminated(energy_norm, action_delta_norm)
        info["energy_norm"] = energy_norm
        info["action_delta_norm"] = action_delta_norm
        terminated = (
            success
            if self.config.terminate_on_success
            else torch.zeros_like(success, dtype=torch.bool)
        )
        truncated = self._elapsed >= self.max_episode_steps
        done = terminated | truncated
        # The transition's task descriptions (matching reward/terminated) are
        # snapshotted before autoreset reassigns done worlds to new episodes.
        info["task_description"] = tuple(self.task_descriptions)
        if bool(done.any()):
            with wp.ScopedDevice(self._wp_device):
                self._write_reset_state(done)
                mjw.forward(self.model, self.data)
                # Settle-independent reset reference (matches reset()): done worlds
                # get identical targets/baselines without settling.
                self._refresh_reset_reference_state(done)
            # Clear previous-action state for reset worlds so a new episode's
            # first action_delta_norm is zero for any first action, not measured
            # against the prior episode's final action. The robot settle that
            # reset() applies is intentionally skipped here: mjw.step advances the
            # whole batch, so settling only-done worlds would advance non-done
            # worlds too. The reset reference above is settle-independent, so it
            # matches reset() exactly; only the robot's first-frame settle transient
            # differs (per-world warmstart left as an optimizer hint).
            self._has_prev_action[done] = False
        obs = self._compute_obs()
        if self._privileged_state is not None:
            info["privileged_state"] = self._privileged_state
        return obs, reward, terminated, truncated, info

    def _finger_geom_mask(self, mjm: mujoco.MjModel, body_id: int, ngeom: int) -> torch.Tensor:
        """Boolean per-geom mask of ``condim==6`` contact geoms on a finger body."""
        mask = torch.zeros(ngeom, dtype=torch.bool, device=self.device)
        for g in range(ngeom):
            if (
                mjm.geom_bodyid[g] == body_id
                and mjm.geom_contype[g] != 0
                and mjm.geom_condim[g] == 6
            ):
                mask[g] = True
        return mask

    def _get_tcp_pose7(self) -> torch.Tensor:
        """Return ``(N, 7)`` TCP pose ``[xyz, wxyz]`` from site position + orientation."""
        pos = self.site_xpos[:, self._tcp_site_id, :]
        quat = _mat_to_quat(self.site_xmat[:, self._tcp_site_id])
        return torch.cat([pos, quat], dim=1)

    def _is_robot_static(self) -> torch.Tensor:
        """Return ``(N,)`` bool: all arm joints below ``static_vel_threshold``."""
        arm_vel = self.qvel.index_select(1, self._arm_dof_adr)
        return (arm_vel.abs() < self.config.robot.static_vel_threshold).all(dim=1)

    def _ensure_grasp_buffers(self) -> None:
        if self._force_buf is not None:
            return
        naconmax = self.data.naconmax
        self._contact_ids = wp.array(
            np.arange(naconmax, dtype=np.int32), dtype=wp.int32, device=self._wp_device
        )
        self._force_buf = wp.zeros(naconmax, dtype=wp.spatial_vector, device=self._wp_device)
        self._force_view = wp.to_torch(self._force_buf)  # (naconmax, 6)

    def _is_grasping(self) -> torch.Tensor:
        """Return ``(N,)`` float in {0, 1}: two-sided finger grasp of the target geom.

        Zero everywhere when no graspable object is registered (``_obj_geom``
        unset), so primitive tasks never trigger grasp logic.
        """
        if self._obj_geom is None:
            return torch.zeros(self.num_envs, device=self.device)
        self._ensure_grasp_buffers()
        force_view = self._force_view
        assert force_view is not None
        with wp.ScopedDevice(self._wp_device):
            mjw.contact_force(self.model, self.data, self._contact_ids, False, self._force_buf)
        nacon = int(self._nacon_view[0])
        return _grasp_from_contacts(
            contact_geom=self._contact_geom_view,
            contact_world=self._contact_world_view,
            normal_force=force_view[:, 0].abs(),
            nacon=nacon,
            obj_geom=self._obj_geom,
            gripper_mask=self._gripper_mask,
            jaw_mask=self._jaw_mask,
            threshold=self.config.robot.grasp_force_threshold,
            num_envs=self.num_envs,
        )

    def _parse_init_qpos(self, options: dict[str, Any] | None) -> torch.Tensor | None:
        """Validate and return the ``options['init_qpos']`` reset override, if any."""
        if options is None or options.get("init_qpos") is None:
            return None
        n_joints = len(SO101_JOINT_NAMES)
        arr = torch.as_tensor(options["init_qpos"], dtype=torch.float32, device=self.device)
        if arr.shape not in {(n_joints,), (self.num_envs, n_joints)}:
            raise ValueError(
                f"init_qpos shape {tuple(arr.shape)} != expected ({n_joints},) "
                f"or ({self.num_envs}, {n_joints})"
            )
        return arr

    def _sample_reset_qpos(self, n: int, init_qpos: torch.Tensor | None) -> torch.Tensor:
        """Return ``(n, 6)`` reset joint targets per the reset contract.

        Priority: explicit ``init_qpos`` (clamped, no noise); else
        ``config.robot.init_pose`` sampled per world with the seeded generator;
        else the rest pose plus per-world uniform ``robot_init_qpos_noise``.
        """
        n_joints = len(SO101_JOINT_NAMES)
        if init_qpos is not None:
            target = init_qpos.expand(n, n_joints) if init_qpos.ndim == 1 else init_qpos
            return torch.clamp(target, self._target_low, self._target_high)
        pose = self.config.robot.resolve_pose()
        if pose is not None:
            low_np, high_np = pose.bounds_rad()
            low = torch.as_tensor(low_np, dtype=torch.float32, device=self.device)
            high = torch.as_tensor(high_np, dtype=torch.float32, device=self.device)
            u = torch.rand((n, n_joints), generator=self._generator, device=self.device)
            return torch.clamp(low + u * (high - low), self._target_low, self._target_high)
        noise = (
            torch.rand((n, n_joints), generator=self._generator, device=self.device) * 2.0 - 1.0
        ) * self.robot_init_qpos_noise
        return torch.clamp(self._rest_qpos + noise, self._target_low, self._target_high)

    def _refresh_reset_reference_state(self, mask: torch.Tensor) -> None:
        """Post-settle hook to refresh task reference state (default no-op)."""

    # Task seams (subclasses implement).
    def _task_reset(self, mask: torch.Tensor) -> None:
        raise NotImplementedError

    def _get_component_data(self, component: object) -> torch.Tensor:
        raise NotImplementedError(
            f"{type(self).__name__} does not support observation component {component!r}"
        )

    def _supported_obs_components(self) -> set[type]:
        """State-component classes this task routes through ``_get_component_data``."""
        return set()

    def _compute_reward_terminated(
        self, energy_norm: torch.Tensor, action_delta_norm: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        raise NotImplementedError
