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

Physics diverges from the MuJoCo backend (see ``so101_nexus.scene``): mujoco_warp
does not support ``implicitfast`` or ``noslip``, so the Warp scene uses the
``implicit`` integrator with no noslip. Camera observations are not supported.
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

from so101_nexus.config import SO101_JOINT_NAMES, ControlMode, EnvironmentConfig
from so101_nexus.observations import (
    CameraObservation,
    EndEffectorPose,
    GraspState,
    JointPositions,
)

# Normalized-delta physical scale (radians), shared with the MuJoCo backend's
# _DELTA_ACTION_SCALE: +/-0.05 for the five arm joints, +/-0.2 for the gripper.
_DELTA_ACTION_SCALE = (0.05, 0.05, 0.05, 0.05, 0.05, 0.2)


def _mat_to_quat(mat: torch.Tensor) -> torch.Tensor:
    """Batched rotation matrix ``(..., 3, 3)`` -> ``wxyz`` quaternion ``(..., 4)``.

    Shepperd's method, computed in float64 for stability and returned in the
    input dtype. Matches ``mujoco.mju_mat2Quat`` up to sign (validated to 1e-9 in
    float64); the result is canonicalized to ``w >= 0``.
    """
    m = mat.to(torch.float64)
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
    quat = torch.where(quat[..., 0:1] < 0.0, -quat, quat)
    return quat.to(mat.dtype)


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
    _VALID_CONTROL_MODES = {
        "pd_joint_pos",
        "pd_joint_delta_pos",
        "pd_joint_target_delta_pos",
    }

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
    ) -> None:
        if control_mode not in self._VALID_CONTROL_MODES:
            raise ValueError(
                f"control_mode must be one of {sorted(self._VALID_CONTROL_MODES)}, "
                f"got {control_mode!r}"
            )
        if config.observations is not None:
            self._validate_obs_components(config.observations)

        self.config = config
        self.control_mode = control_mode
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
        self._tcp_site_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")

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
        else:
            self.single_action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(n_joints,), dtype=np.float32
            )
            self._action_low = torch.full((n_joints,), -1.0, device=self.device)
            self._action_high = torch.full((n_joints,), 1.0, device=self.device)
        self.single_observation_space = spaces.Box(
            -np.inf, np.inf, shape=(self._obs_dim(),), dtype=np.float32
        )
        self.num_envs = num_envs
        self.action_space = batch_space(self.single_action_space, num_envs)
        self.observation_space = batch_space(self.single_observation_space, num_envs)

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
        self._step_graph = None
        self._capture_step_graph()

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
        ``EndEffectorPose``, ``GraspState``) centrally, mirroring the MuJoCo base,
        and delegates task-specific components to ``_get_component_data`` (a
        subclass declares those via ``_supported_obs_components``). Camera
        components are unsupported on Warp. Anything else raises here rather than
        at the first reset, so the error names the unsupported component upfront.
        """
        supported = {
            JointPositions,
            EndEffectorPose,
            GraspState,
            *self._supported_obs_components(),
        }
        for comp in observations:
            if isinstance(comp, CameraObservation):
                raise NotImplementedError("Warp backend does not support camera observations")
            if not isinstance(comp, tuple(supported)):
                raise NotImplementedError(
                    f"{type(self).__name__} does not support observation "
                    f"component {comp!r} on the Warp backend"
                )

    def _obs_dim(self) -> int:
        if self.config.observations is None:
            raise RuntimeError("config.observations must be set")
        return sum(c.size for c in self.config.observations if c.size > 0)

    def _joint_qpos(self) -> torch.Tensor:
        return self.qpos.index_select(1, self._qpos_adr)

    def _tcp_pos(self) -> torch.Tensor:
        return self.site_xpos[:, self._tcp_site_id, :]

    def _compute_obs(self) -> torch.Tensor:
        if self.config.observations is None:
            raise RuntimeError("config.observations must be set")
        parts: list[torch.Tensor] = []
        for comp in self.config.observations:
            if isinstance(comp, JointPositions):
                parts.append(self._joint_qpos())
            elif isinstance(comp, EndEffectorPose):
                parts.append(self._get_tcp_pose7())
            elif isinstance(comp, GraspState):
                parts.append(self._is_grasping().unsqueeze(1))
            elif isinstance(comp, CameraObservation):
                raise NotImplementedError("Warp backend does not support camera observations")
            else:
                parts.append(self._get_component_data(comp))
        return torch.cat(parts, dim=1).to(torch.float32)

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

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[torch.Tensor, dict]:
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
        return self._compute_obs(), {}

    def close(self, **kwargs: Any) -> None:
        """No-op: Warp device memory is released when the env is garbage-collected."""

    def _action_to_ctrl(self, action: torch.Tensor) -> torch.Tensor:
        if self.control_mode == "pd_joint_pos":
            return torch.clamp(action, self._target_low, self._target_high)
        delta = action * self._delta_scale
        if self.control_mode == "pd_joint_delta_pos":
            return torch.clamp(self._joint_qpos() + delta, self._target_low, self._target_high)
        self._prev_target = torch.clamp(
            self._prev_target + delta, self._target_low, self._target_high
        )
        return self._prev_target

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
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
        terminated = success
        truncated = self._elapsed >= self.max_episode_steps
        done = terminated | truncated
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
        return self._compute_obs(), reward, terminated, truncated, info

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
