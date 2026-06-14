"""ManiSkill base environment for SO101-Nexus simulation tasks."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import sapien
import sapien.render
import torch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
from sapien.render import RenderBodyComponent
from transforms3d.euler import euler2quat

from so101_nexus_core.camera_utils import compute_overhead_eye_target
from so101_nexus_core.constants import sample_color
from so101_nexus_core.observations import (
    EndEffectorPose,
    GraspState,
    JointPositions,
    OverheadCamera,
    WristCamera,
    _CameraObservation,
)

if TYPE_CHECKING:
    from mani_skill.agents.robots.so100.so_100 import SO100

    from so101_nexus_core.config import EnvironmentConfig
    from so101_nexus_maniskill.so101_agent import SO101

logger = logging.getLogger(__name__)


class SO101NexusManiSkillBaseEnv(BaseEnv):
    """Shared ManiSkill base class for SO101-Nexus tasks."""

    SUPPORTED_ROBOTS = ["so100", "so101"]
    agent: SO100 | SO101

    def _setup_base(
        self,
        *,
        config: EnvironmentConfig,
        robot_uids: str,
        robot_cfgs: dict[str, dict[str, Any]],
        robot_init_qpos_noise: float | None = None,
    ) -> None:
        if robot_uids not in robot_cfgs:
            raise ValueError(f"robot_uids must be one of {list(robot_cfgs)}, got {robot_uids!r}")

        self.config = config
        self.robot_init_qpos_noise = (
            robot_init_qpos_noise
            if robot_init_qpos_noise is not None
            else config.robot_init_qpos_noise
        )
        self._robot_cfg = robot_cfgs[robot_uids]
        self._initial_obj_z: torch.Tensor | None = None
        self._init_qpos_clamp_warned = False
        # Previous public policy action per env row, used for the action-smoothness
        # penalty. ``_has_prev_action`` is False for rows that have been reset but
        # not yet stepped, so the first step after a reset reports a delta of 0.
        self._prev_action: torch.Tensor | None = None
        self._has_prev_action: torch.Tensor | None = None

        # Detect camera observation components
        self._wrist_cam_component: WristCamera | None = None
        self._overhead_cam_component: OverheadCamera | None = None
        if config.observations is not None:
            for comp in config.observations:
                if isinstance(comp, WristCamera):
                    self._wrist_cam_component = comp
                elif isinstance(comp, OverheadCamera):
                    self._overhead_cam_component = comp

    def _default_reconfiguration_freq(self) -> int:
        """Return the default reconfiguration frequency based on camera components.

        Wrist camera needs scene reconfiguration every episode
        so that the wrist camera updates correctly.
        """
        return 1 if self._wrist_cam_component is not None else 0

    # Tensor equivalent of so101_nexus_core.rewards.reach_progress
    def _reach_progress(self, dist: torch.Tensor) -> torch.Tensor:
        """Tanh-shaped progress in [0, 1]. See ``so101_nexus_core.rewards.reach_progress``."""
        return 1.0 - torch.tanh(self.config.reward.tanh_shaping_scale * dist)

    # Tensor equivalent of so101_nexus_core.config.RewardConfig.apply_penalties
    def _apply_penalties_tensor(
        self,
        base: torch.Tensor,
        action_delta_norm: float | torch.Tensor,
        energy_norm: float | torch.Tensor,
    ) -> torch.Tensor:
        """Subtract action/energy penalties. See ``RewardConfig.apply_penalties``."""
        cfg = self.config.reward
        return (
            base - cfg.action_delta_penalty * action_delta_norm - cfg.energy_penalty * energy_norm
        )

    # Tensor equivalent of so101_nexus_core.config.RewardConfig.compute
    def _assemble_normalized_reward(
        self,
        *,
        reach_progress: torch.Tensor,
        is_grasped: torch.Tensor,
        task_progress: torch.Tensor,
        is_complete: torch.Tensor,
        action_delta_norm: float | torch.Tensor = 0.0,
        energy_norm: float | torch.Tensor = 0.0,
    ) -> torch.Tensor:
        """Assemble weighted reward. See ``RewardConfig.compute`` for scalar equivalent."""
        cfg = self.config.reward
        base = (
            cfg.reaching * reach_progress
            + cfg.grasping * is_grasped
            + cfg.task_objective * task_progress
            + cfg.completion_bonus * is_complete
        )
        return self._apply_penalties_tensor(base, action_delta_norm, energy_norm)

    def get_reward(
        self, obs: Any, action: torch.Tensor, info: dict[str, Any]
    ) -> torch.Tensor:
        """Stamp penalty norms once per step, then delegate to ManiSkill's dispatch.

        ``get_reward`` is the single hook ManiSkill calls exactly once per step
        (after ``get_info``/``evaluate`` and ``get_obs``) for every reward mode,
        and is the only place the public action and the info dict are both
        available. Stamping here (rather than inside ``compute_dense_reward``)
        guarantees ``info["energy_norm"]``/``info["action_delta_norm"]`` are
        written and ``_prev_action`` advances under "sparse"/"none"/"dense"
        alike, matching the MuJoCo backend which stamps unconditionally in
        ``step``. The reward computations then read the precomputed norms from
        ``info`` instead of recomputing or advancing state.
        """
        self._stamp_action_penalty_norms(action, info)
        return super().get_reward(obs=obs, action=action, info=info)

    def _stamp_action_penalty_norms(
        self, action: torch.Tensor, info: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute batched energy and action-delta norms, stamp ``info``, update state.

        ``energy_norm`` is the per-row L2 norm of the action. ``action_delta_norm``
        is the per-row L2 norm of the difference from the previous action, and is 0
        for rows whose first step has not yet occurred since the last reset. Both
        are written into ``info`` so they reach the user-facing info dict, then the
        stored previous action is advanced. Called exactly once per step from
        ``get_reward``. Norms use the public action as received by ``step`` (before
        any controller clipping), matching the MuJoCo backend convention so the
        penalty is comparable across backends.
        """
        energy_norm = torch.linalg.norm(action, dim=-1)
        has_prev_action = getattr(self, "_has_prev_action", None)
        if has_prev_action is None:
            # State buffers were never allocated (e.g. a unit test instantiating
            # the env via object.__new__); fall back to a zero delta.
            action_delta_norm = torch.zeros_like(energy_norm)
        else:
            prev_action = getattr(self, "_prev_action", None)
            if prev_action is None:
                prev_action = torch.zeros_like(action)
            raw_delta = torch.linalg.norm(action - prev_action, dim=-1)
            action_delta_norm = torch.where(has_prev_action, raw_delta, torch.zeros_like(raw_delta))
            # Advance stored state: every stepped row now has a valid prev action.
            self._prev_action = action.detach().clone()
            self._has_prev_action = torch.ones_like(has_prev_action)
        info["energy_norm"] = energy_norm
        info["action_delta_norm"] = action_delta_norm
        return action_delta_norm, energy_norm

    @property
    def _default_sim_config(self) -> SimConfig:
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                max_rigid_contact_count=2**20, max_rigid_patch_count=2**19
            )
        )

    @property
    def _default_sensor_configs(self) -> list[CameraConfig]:
        cfg = self._robot_cfg
        configs: list[CameraConfig] = []

        # Wrist camera: driven by WristCamera observation component
        if self._wrist_cam_component is not None:
            wc = self._wrist_cam_component
            w = wc.width
            h = wc.height
            mount_link = self.agent.robot.links_map[cfg["wrist_camera_mount_link"]]

            if self.agent.uid == "so101":
                # One source of truth with the MuJoCo backend: build the wrist
                # camera pose from the WristCamera component (same fields
                # _randomize_wrist_camera uses), then convert the MuJoCo camera
                # convention (looks along -z, +y up) to SAPIEN's (+x). x is
                # noise-only (centered at 0), matching MuJoCo's cam_pos0.
                # Sample with the seeded episode RNG (set before reconfigure and
                # re-seeded after), so env.reset(seed=...) reproduces the camera,
                # matching the MuJoCo backend's self.np_random usage. Falls back
                # to the global RNG only if accessed before the first seeded reset.
                from transforms3d.quaternions import qmult

                from so101_nexus_maniskill import menagerie_constants as mc

                rng = (
                    self._episode_rng
                    if getattr(self, "_episode_rng", None) is not None
                    else np.random
                )
                px = rng.uniform(-wc.pos_x_noise, wc.pos_x_noise)
                py = wc.pos_y_center + rng.uniform(-wc.pos_y_noise, wc.pos_y_noise)
                pz = wc.pos_z_center + rng.uniform(-wc.pos_z_noise, wc.pos_z_noise)
                p = [px, py, pz]
                pitch_lo, pitch_hi = wc.pitch_rad_range
                pitch = rng.uniform(pitch_lo, pitch_hi)
                q_mujoco = euler2quat(pitch, 0.0, 0.0, axes="sxyz")
                # Order pinned by test_wrist_camera_world_pose_matches_mujoco_backend.
                q = qmult(q_mujoco, mc.MJ_TO_SAPIEN_CAMERA_QUAT)
                fov_lo, fov_hi = wc.fov_rad_range
                fov = rng.uniform(fov_lo, fov_hi)
            else:
                # so100: existing preset-driven path (unchanged).
                pos_c = cfg["wrist_cam_pos_center"]
                pos_n = cfg["wrist_cam_pos_noise"]
                eul_c = cfg["wrist_cam_euler_center"]
                eul_n = cfg["wrist_cam_euler_noise"]
                fov_lo, fov_hi = cfg["wrist_cam_fov_range"]
                p = [c + np.random.uniform(-n, n) for c, n in zip(pos_c, pos_n, strict=True)]
                e = [c + np.random.uniform(-n, n) for c, n in zip(eul_c, eul_n, strict=True)]
                q = euler2quat(*e, axes="sxyz")
                fov = np.random.uniform(fov_lo, fov_hi)

            configs.append(
                CameraConfig(
                    "wrist_camera",
                    Pose.create_from_pq(
                        p=torch.tensor(p, dtype=torch.float32),
                        q=torch.tensor(q, dtype=torch.float32),
                    ),
                    w,
                    h,
                    fov,
                    0.01,
                    100,
                    mount=mount_link,
                )
            )

        # Overhead camera: driven by OverheadCamera observation component
        if self._overhead_cam_component is not None:
            oc = self._overhead_cam_component
            eye, target = compute_overhead_eye_target(
                spawn_center=self.config.spawn_center,
                spawn_max_radius=self.config.spawn_max_radius,
                fov_deg=oc.fov_deg,
                aspect=oc.width / oc.height,
            )
            pose = sapien_utils.look_at(eye, target, up=(1, 0, 0))
            configs.append(
                CameraConfig(
                    "overhead_camera",
                    pose,
                    oc.width,
                    oc.height,
                    float(np.radians(oc.fov_deg)),
                    0.01,
                    100,
                )
            )

        return configs

    @property
    def _default_human_render_camera_configs(self) -> CameraConfig:
        rw = self.config.render.width
        rh = self.config.render.height
        eye, target = compute_overhead_eye_target(
            spawn_center=self.config.spawn_center,
            spawn_max_radius=self.config.spawn_max_radius,
            aspect=rw / rh,
        )
        # up=(1,0,0) so +X (robot forward) points up in the image.
        pose = sapien_utils.look_at(eye, target, up=(1, 0, 0))
        return CameraConfig("render_camera", pose, rw, rh, 1, 0.01, 100)

    def _load_agent(
        self,
        options: dict,
        initial_agent_poses: sapien.Pose | Pose | None = None,
        build_separate: bool = False,
    ) -> None:
        pose = initial_agent_poses or sapien.Pose(p=[0, 0, 0], q=self._robot_cfg["base_quat"])
        super()._load_agent(options, pose, build_separate)

    def _load_lighting(self, options: dict) -> None:
        self.scene.set_ambient_light([0.3, 0.3, 0.3])
        self.scene.add_directional_light(
            [1, 1, -1],
            color=[1, 1, 1],
            shadow=self.enable_shadow,
            shadow_scale=5,
            shadow_map_size=2048,
        )
        self.scene.add_directional_light([0, 0, -1], color=[1, 1, 1])

    def _after_reconfigure(self, options: dict) -> None:
        self._initial_obj_z = torch.zeros(self.num_envs, device=self.device)
        # _prev_action is allocated lazily on the first step (the action shape is
        # not known here). _has_prev_action gates the action-delta penalty so the
        # first step after each reset reports a delta of 0.
        self._prev_action = None
        self._has_prev_action = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _build_ground(self) -> None:
        ground_builder = self.scene.create_actor_builder()
        ground_builder.add_plane_collision(
            sapien.Pose(p=[0, 0, 0], q=[0.7071068, 0, -0.7071068, 0])
        )
        ground_builder.initial_pose = sapien.Pose(p=[0, 0, 0])
        if self.scene.parallel_in_single_scene:
            ground_builder.set_scene_idxs([0])
        ground = ground_builder.build_static(name="ground")

        if not self.scene.can_render():
            return

        floor_half = 50
        verts = np.array(
            [
                [-floor_half, -floor_half, 0],
                [floor_half, -floor_half, 0],
                [floor_half, floor_half, 0],
                [-floor_half, floor_half, 0],
            ],
            dtype=np.float32,
        )
        normals = np.tile([0, 0, 1], (4, 1)).astype(np.float32)
        tris = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        uvs = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
        mat = sapien.render.RenderMaterial()
        mat.base_color = sample_color(self.config.ground_colors)
        shape = sapien.render.RenderShapeTriangleMesh(
            vertices=verts,
            triangles=tris,
            normals=normals,
            uvs=uvs,
            material=mat,
        )
        for obj in ground._objs:
            comp = sapien.render.RenderBodyComponent()
            comp.attach(shape)
            obj.add_component(comp)

    def _apply_robot_color_if_needed(self) -> None:
        color = sample_color(self.config.robot_colors)
        for link in self.agent.robot.links:
            for obj in link._objs:
                render_body: RenderBodyComponent = obj.entity.find_component_by_type(
                    RenderBodyComponent
                )
                if render_body is None:
                    continue
                for render_shape in render_body.render_shapes:
                    for part in render_shape.parts:
                        part.material.set_base_color(color)

    def _clamp_to_qlimits(self, qpos: torch.Tensor, env_idx: torch.Tensor) -> torch.Tensor:
        """Clamp the reset qpos to the joint limits of the reset envs.

        ``qpos`` has shape ``(len(env_idx), dof)``; ``get_qlimits()`` has shape
        ``(num_envs, dof, 2)``. Slice by ``env_idx`` FIRST so a partial reset
        (e.g. ``env_idx=[0]`` in a multi-env scene) clamps against the right
        rows instead of broadcasting one reset qpos against every env row.
        SAPIEN does not clamp set_qpos; controller bounds equal the joint
        limits, so clamping to qlimits mirrors the MuJoCo backend's
        ctrlrange/joint-range clamp.
        """
        qlimits = self.agent.robot.get_qlimits()[env_idx]  # (len(env_idx), dof, 2)
        return torch.clamp(qpos, qlimits[..., 0], qlimits[..., 1])

    def _reset_robot(self, env_idx: torch.Tensor, options: dict | None = None) -> None:
        # Mark the reset rows as having no previous action so the first step after
        # this (full or partial) reset reports an action_delta_norm of 0.
        if self._has_prev_action is not None:
            self._has_prev_action[env_idx] = False
        b = len(env_idx)
        init_qpos = self._init_qpos_from_options(options, env_idx)
        if init_qpos is not None:
            qpos = init_qpos
        elif (pose := self.config.robot.resolve_pose()) is not None:
            seed = int(torch.randint(0, 2**31, (1,)).item())
            np_rng = np.random.default_rng(seed)
            samples = np.array([pose.sample_rad(np_rng) for _ in range(b)], dtype=np.float32)
            qpos = torch.tensor(samples, dtype=torch.float32, device=self.device)
        else:
            # Default rest path: source of truth is config.robot.rest_qpos_rad,
            # matching the MuJoCo backend (honors a user RobotConfig override).
            rest = torch.tensor(
                self.config.robot.rest_qpos_rad, dtype=torch.float32, device=self.device
            )
            qpos = rest.unsqueeze(0).expand(b, -1).clone()
            noise = (torch.rand_like(qpos) * 2 - 1) * self.robot_init_qpos_noise
            qpos = qpos + noise
        # SAPIEN does not clamp set_qpos; clamp every source (incl. post-noise).
        qpos = self._clamp_to_qlimits(qpos, env_idx)
        self.agent.reset(qpos)
        # BaseAgent.reset sets qpos/qvel/qf but NOT the PhysX drive targets, and
        # ManiSkill only calls controller.reset() after _initialize_episode
        # returns (after _settle_after_reset has already stepped). Critically,
        # PDJointPosController.reset() only updates the controller's private
        # _start_qpos/_target_qpos; it never writes the PhysX drive targets
        # (verified in mani_skill 3.0.1 pd_joint_pos.py: reset() does not call
        # set_drive_targets). So reset the controller AND explicitly write the
        # drive targets to the clamped reset qpos, so settle frames hold the
        # reset pose instead of pulling toward stale targets from the previous
        # episode. set_drive_targets uses scene._reset_mask internally, so the
        # (len(env_idx), dof) qpos is the correct shape for partial resets.
        self.agent.controller.reset()
        self.agent.controller.set_drive_targets(qpos)
        self.agent.robot.set_pose(sapien.Pose(p=[0, 0, 0], q=self._robot_cfg["base_quat"]))

    def _init_qpos_from_options(
        self, options: dict | None, env_idx: torch.Tensor
    ) -> torch.Tensor | None:
        """Return a validated, qlimit-clamped reset qpos from options['init_qpos']."""
        if options is None or "init_qpos" not in options:
            return None

        batch_size = len(env_idx)
        expected = len(self.agent.keyframes["rest"].qpos)
        raw_qpos = torch.as_tensor(options["init_qpos"], dtype=torch.float32, device=self.device)
        if raw_qpos.shape == (expected,):
            qpos = raw_qpos.unsqueeze(0).expand(batch_size, -1).clone()
        elif raw_qpos.shape == (batch_size, expected):
            qpos = raw_qpos.clone()
        else:
            raise ValueError(
                f"init_qpos shape {tuple(raw_qpos.shape)} != expected "
                f"({expected},) or ({batch_size}, {expected})"
            )
        clamped = self._clamp_to_qlimits(qpos, env_idx)
        if not torch.equal(clamped, qpos) and not self._init_qpos_clamp_warned:
            logger.warning(
                "init_qpos clamped to joint limits for at least one joint; "
                "further init_qpos clamps on this env will be silent."
            )
            self._init_qpos_clamp_warned = True
        return clamped

    def _store_initial_obj_z(self, env_idx: torch.Tensor, z: torch.Tensor) -> None:
        if self._initial_obj_z is None:
            self._initial_obj_z = torch.zeros(self.num_envs, device=self.device)
        self._initial_obj_z[env_idx] = z

    def _inactive_env_idx(self, env_idx: torch.Tensor) -> torch.Tensor:
        """Return vectorized env rows that are not part of the current reset."""
        mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        mask[env_idx] = False
        return torch.arange(self.num_envs, device=self.device)[mask]

    def _settle_after_reset(self, env_idx: torch.Tensor, *, min_frames: int = 0) -> None:
        """Advance configured no-op frames after reset while preserving inactive rows."""
        frames = max(self.config.reset_settle_frames, min_frames)
        if frames == 0:
            return

        inactive_idx = self._inactive_env_idx(env_idx)
        pre_settle_state = self.get_state_dict() if len(inactive_idx) else None
        for _ in range(frames):
            self.scene.step()
        if pre_settle_state is not None:
            # Snapshot covers all envs; env_idx selects only inactive rows to restore.
            self.set_state_dict(pre_settle_state, inactive_idx)

    def _refresh_reset_reference_state(self, env_idx: torch.Tensor) -> None:
        """Refresh task reference state after reset settling."""

    def _build_obs_extra_from_components(self, info: dict) -> dict[str, torch.Tensor]:
        """Build obs_extra dict from observation components.

        ManiSkill automatically includes agent qpos/qvel. This method adds
        task-specific components from config.observations.
        """
        obs: dict[str, torch.Tensor] = {}
        if self.config.observations is None:
            return obs
        for comp in self.config.observations:
            if isinstance(comp, JointPositions):
                continue  # ManiSkill includes qpos automatically
            if isinstance(comp, _CameraObservation):
                continue  # Handled via _default_sensor_configs
            if isinstance(comp, EndEffectorPose):
                obs["tcp_pose"] = self.agent.tcp_pose.raw_pose
            elif isinstance(comp, GraspState):
                obs["is_grasped"] = info.get(
                    "is_grasped", torch.zeros(self.num_envs, device=self.device)
                )
            else:
                self._add_component_obs(obs, comp, info)
        return obs

    def _add_component_obs(
        self, obs: dict[str, torch.Tensor], component: object, info: dict
    ) -> None:
        """Add a task-specific component to obs dict. Subclasses override."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support observation component {component!r}"
        )

    def compute_normalized_dense_reward(
        self,
        obs: dict[str, torch.Tensor] | torch.Tensor,
        action: torch.Tensor,
        info: dict[str, Any],
    ) -> torch.Tensor:
        """Delegate to compute_dense_reward (reward is already normalized)."""
        return self.compute_dense_reward(obs=obs, action=action, info=info)


def register_robot_variant(
    *,
    class_name: str,
    env_id: str,
    base_cls: type,
    robot_uid: str,
    max_episode_steps: int,
    caller_globals: dict,
) -> type:
    """Create and register a robot-specific environment variant.

    Parameters
    ----------
    class_name:
        Name for the generated class.
    env_id:
        Gymnasium environment ID to register.
    base_cls:
        Base environment class to subclass.
    robot_uid:
        Robot identifier (``"so100"`` or ``"so101"``).
    max_episode_steps:
        Maximum episode length for registration.
    caller_globals:
        The calling module's ``globals()`` dict so the class is
        injected into the correct namespace.
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("robot_uids", robot_uid)
        base_cls.__init__(self, *args, **kwargs)

    cls = type(class_name, (base_cls,), {"__init__": __init__})
    cls = register_env(env_id, max_episode_steps=max_episode_steps)(cls)
    caller_globals[class_name] = cls
    return cls
