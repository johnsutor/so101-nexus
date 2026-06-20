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
from so101_nexus.observations import CameraObservation, JointPositions

# Normalized-delta physical scale (radians), shared with the MuJoCo backend's
# _DELTA_ACTION_SCALE: +/-0.05 for the five arm joints, +/-0.2 for the gripper.
_DELTA_ACTION_SCALE = (0.05, 0.05, 0.05, 0.05, 0.05, 0.2)


class SO101NexusWarpVectorEnv(VectorEnv):
    """Shared GPU-batched Warp base class for SO101-Nexus tasks."""

    metadata = {"render_modes": [], "autoreset_mode": AutoresetMode.SAME_STEP}
    _N_SUBSTEPS = 4
    _VALID_CONTROL_MODES = {
        "pd_joint_pos",
        "pd_joint_delta_pos",
        "pd_joint_target_delta_pos",
    }

    def __init__(
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
        if config.observations is not None and any(
            isinstance(c, CameraObservation) for c in config.observations
        ):
            raise NotImplementedError("Warp backend does not support camera observations")

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
            # nconmax/njmax default to None (mujoco_warp auto-sizes per world).
            # Step 5 verifies a rollout does not overflow; set explicitly if it does.
            self.data = mjw.put_data(mjm, mjd, nworld=num_envs, nconmax=nconmax, njmax=njmax)

        # Zero-copy torch views. NEVER rebind these; mutate in place only.
        self.qpos = wp.to_torch(self.data.qpos)  # (N, nq)
        self.qvel = wp.to_torch(self.data.qvel)  # (N, nv)
        self.ctrl = wp.to_torch(self.data.ctrl)  # (N, nu)
        self.site_xpos = wp.to_torch(self.data.site_xpos)  # (N, nsite, 3)

        joint_ids = [
            mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, n) for n in SO101_JOINT_NAMES
        ]
        act_ids = [
            mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in SO101_JOINT_NAMES
        ]
        self._qpos_adr = torch.as_tensor(
            [mjm.jnt_qposadr[j] for j in joint_ids], device=self.device
        )
        self._dof_adr = torch.as_tensor(
            [mjm.jnt_dofadr[j] for j in joint_ids], device=self.device
        )
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
        self._prev_target = self._rest_qpos.expand(num_envs, n_joints).clone()

    def _obs_dim(self) -> int:
        if self.config.observations is None:
            raise RuntimeError("config.observations must be set")
        return sum(c.size for c in self.config.observations if c.size > 0)

    def _joint_qpos(self) -> torch.Tensor:
        return self.qpos.index_select(1, self._qpos_adr)

    def _tcp_pos(self) -> torch.Tensor:
        return self.site_xpos[:, self._tcp_site_id, :]

    def _compute_obs(self) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        for comp in self.config.observations:
            if isinstance(comp, JointPositions):
                parts.append(self._joint_qpos())
            elif isinstance(comp, CameraObservation):
                raise NotImplementedError("Warp backend does not support camera observations")
            else:
                parts.append(self._get_component_data(comp))
        return torch.cat(parts, dim=1).to(torch.float32)

    def _write_reset_state(self, mask: torch.Tensor) -> None:
        idx = mask.nonzero(as_tuple=True)[0]
        n = int(idx.numel())
        if n == 0:
            return
        noise = (
            torch.rand(
                (n, len(SO101_JOINT_NAMES)), generator=self._generator, device=self.device
            )
            * 2.0
            - 1.0
        ) * self.robot_init_qpos_noise
        target = torch.clamp(self._rest_qpos + noise, self._target_low, self._target_high)
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
        self._prev_action = None
        mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        with wp.ScopedDevice(self._wp_device):
            self._write_reset_state(mask)
            mjw.forward(self.model, self.data)
            for _ in range(self.config.reset_settle_frames):
                for _ in range(self._N_SUBSTEPS):
                    mjw.step(self.model, self.data)
        return self._compute_obs(), {}

    def close(self) -> None:
        """No-op: Warp device memory is released when the env is garbage-collected."""

    # Task seams (subclasses implement).
    def _task_reset(self, mask: torch.Tensor) -> None:
        raise NotImplementedError

    def _get_component_data(self, component: object) -> torch.Tensor:
        raise NotImplementedError(
            f"{type(self).__name__} does not support observation component {component!r}"
        )

    def _compute_reward_terminated(
        self, energy_norm: torch.Tensor, action_delta_norm: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        raise NotImplementedError
