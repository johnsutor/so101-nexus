"""ManiSkill pick-and-place task environment for SO101-Nexus."""

from typing import Any, ClassVar

import sapien
import torch
from mani_skill.envs.utils.randomization.pose import random_quaternions
from mani_skill.utils.building import actors
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose

from so101_nexus_core.config import PickAndPlaceConfig
from so101_nexus_core.constants import sample_color
from so101_nexus_core.observations import (
    ObjectOffset,
    ObjectPose,
    TargetOffset,
    TargetPosition,
)
from so101_nexus_core.robot_presets import build_maniskill_robot_configs
from so101_nexus_maniskill.base_env import SO101NexusManiSkillBaseEnv, register_robot_variant

_DEFAULT_CONFIG = PickAndPlaceConfig()
PICK_AND_PLACE_CONFIGS: dict[str, dict] = build_maniskill_robot_configs(config=_DEFAULT_CONFIG)


def _sample_polar_arc_xy(
    *,
    rows: int,
    min_r: float,
    max_r: float,
    angle_half: float,
    center: tuple[float, float],
    device: torch.device | str,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample ``rows`` XY positions uniformly in a polar arc around ``center``."""
    cx, cy = center
    r = min_r + torch.rand(rows, device=device, generator=generator) * (max_r - min_r)
    theta = (torch.rand(rows, device=device, generator=generator) * 2 - 1) * angle_half
    xy = torch.zeros((rows, 2), device=device)
    xy[:, 0] = cx + r * torch.cos(theta)
    xy[:, 1] = cy + r * torch.sin(theta)
    return xy


def sample_cube_xy_separated_from_target(
    *,
    target_xy: torch.Tensor,
    min_r: float,
    max_r: float,
    angle_half: float,
    min_separation: float,
    center: tuple[float, float],
    device: torch.device | str,
    max_attempts: int = 100,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample per-env cube XY in a polar arc, separated from a fixed target XY.

    For each environment row, an XY position is sampled in the polar arc and
    rejected if it lies within ``min_separation`` of that row's ``target_xy``.
    Only rows that still violate the constraint are resampled, up to
    ``max_attempts`` times, after which the last sample is accepted as best
    effort. This mirrors the per-row mask semantics of
    ``spawn_utils.sample_separated_positions_torch`` for the single-object case.

    Parameters
    ----------
    target_xy : torch.Tensor
        Shape ``(num_envs, 2)`` fixed target XY positions to stay clear of.
    min_r, max_r : float
        Radial bounds from ``center``, in metres.
    angle_half : float
        Half-angle of the arc in radians; samples are drawn from
        ``[-angle_half, angle_half]``.
    min_separation : float
        Minimum XY distance between the cube and its target.
    center : tuple[float, float]
        XY offset applied to all sampled positions.
    device : torch.device or str
        Device on which to allocate tensors.
    max_attempts : int, optional
        Maximum resampling attempts before accepting best effort.
    generator : torch.Generator, optional
        Optional torch RNG for reproducible sampling.

    Returns
    -------
    torch.Tensor
        Shape ``(num_envs, 2)`` cube XY positions per environment.
    """
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")

    num_envs = int(target_xy.shape[0])
    sampler = _sample_polar_arc_xy

    xy = sampler(
        rows=num_envs,
        min_r=min_r,
        max_r=max_r,
        angle_half=angle_half,
        center=center,
        device=device,
        generator=generator,
    )
    for _ in range(max_attempts):
        dists = torch.linalg.norm(xy - target_xy, dim=1)  # (num_envs,)
        invalid = dists < min_separation  # (num_envs,)
        # bool(invalid.any()) forces a device-to-host sync per iteration
        # (bounded by max_attempts), an accepted tradeoff for this rejection
        # loop. On exhausting max_attempts the loop falls through and the last
        # sample is returned even if still too close (silent best-effort).
        if not bool(invalid.any()):
            break
        n_bad = int(invalid.sum())
        # Resample ONLY the invalid rows; valid rows are left untouched.
        xy[invalid] = sampler(
            rows=n_bad,
            min_r=min_r,
            max_r=max_r,
            angle_half=angle_half,
            center=center,
            device=device,
            generator=generator,
        )

    return xy


class PickAndPlaceEnv(SO101NexusManiSkillBaseEnv):
    """Pick-and-place environment with a visible coloured target disc on the ground."""

    config: PickAndPlaceConfig
    default_config_cls: ClassVar[type[PickAndPlaceConfig]] = PickAndPlaceConfig

    def __init__(
        self,
        *args,
        config: PickAndPlaceConfig | None = None,
        robot_uids: str = "so100",
        num_envs: int = 1,
        reconfiguration_freq: int | None = None,
        robot_init_qpos_noise: float | None = None,
        **kwargs,
    ):
        if config is None:
            config = PickAndPlaceConfig()
        self.cube_colors = config.cube_colors
        self.target_colors = config.target_colors
        self.cube_half_size = config.cube_half_size
        self.target_disc_radius = config.target_disc_radius

        robot_cfgs = build_maniskill_robot_configs(config=config)

        self._setup_base(
            config=config,
            robot_uids=robot_uids,
            robot_cfgs=robot_cfgs,
            robot_init_qpos_noise=robot_init_qpos_noise,
        )
        self.task_description = self.config.task_description

        super().__init__(
            *args,
            robot_uids=robot_uids,
            reconfiguration_freq=(
                reconfiguration_freq
                if reconfiguration_freq is not None
                else self._default_reconfiguration_freq()
            ),
            num_envs=num_envs,
            **kwargs,
        )

    def _load_scene(self, options: dict) -> None:
        self._build_ground()

        objs: list[Actor] = []
        for i in range(self.num_envs):
            cube = actors.build_cube(
                self.scene,
                half_size=self.cube_half_size,
                color=sample_color(self.cube_colors),
                name=f"cube-{i}",
                body_type="dynamic",
                scene_idxs=[i],
                initial_pose=sapien.Pose(
                    p=[0, 0, self.cube_half_size],
                    q=[1, 0, 0, 0],
                ),
            )
            objs.append(cube)
            self.remove_from_state_dict_registry(cube)
        self.obj = Actor.merge(objs, name="cube")
        self.add_to_state_dict_registry(self.obj)

        targets: list[Actor] = []
        for i in range(self.num_envs):
            builder = self.scene.create_actor_builder()
            builder.add_cylinder_visual(
                radius=self.target_disc_radius,
                pose=sapien.Pose(p=[0, 0, 0.001], q=[0, 0.7071068, 0.7071068, 0]),
                half_length=0.001,
                material=sapien.render.RenderMaterial(base_color=sample_color(self.target_colors)),
            )
            builder.initial_pose = sapien.Pose(p=[0, 0, 0], q=[0.7071068, 0.7071068, 0, 0])
            builder.set_scene_idxs([i])
            target = builder.build_kinematic(name=f"target_site-{i}")
            targets.append(target)
            self.remove_from_state_dict_registry(target)
        self.target_site = Actor.merge(targets, name="target_site")
        self.add_to_state_dict_registry(self.target_site)
        self._apply_robot_color_if_needed()

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict) -> None:
        with torch.device(self.device):
            b = len(env_idx)
            self._reset_robot(env_idx, options)

            cfg = self._robot_cfg
            min_r = cfg["spawn_min_radius"]
            max_r = cfg["spawn_max_radius"]
            angle_half = cfg["spawn_angle_half_range"]
            cx, cy = cfg["cube_spawn_center"]

            r_t = min_r + torch.rand(b, device=self.device) * (max_r - min_r)
            theta_t = (torch.rand(b, device=self.device) * 2 - 1) * angle_half
            target_xyz = torch.zeros((b, 3), device=self.device)
            target_xyz[:, 0] = cx + r_t * torch.cos(theta_t)
            target_xyz[:, 1] = cy + r_t * torch.sin(theta_t)
            target_xyz[:, 2] = 0.001
            target_q = torch.tensor([[0.7071068, 0.7071068, 0.0, 0.0]], device=self.device).expand(
                b, -1
            )
            self.target_site.set_pose(Pose.create_from_pq(p=target_xyz, q=target_q))

            xyz = torch.zeros((b, 3), device=self.device)
            # Resample only the rows whose cube falls within
            # min_cube_target_separation of their target (per-row mask), instead
            # of forcing every row to resample when any single row fails.
            xyz[:, :2] = sample_cube_xy_separated_from_target(
                target_xy=target_xyz[:, :2],
                min_r=min_r,
                max_r=max_r,
                angle_half=angle_half,
                min_separation=self.config.min_cube_target_separation,
                center=(cx, cy),
                device=self.device,
            )
            xyz[:, 2] = self.cube_half_size
            qs = random_quaternions(b, lock_x=True, lock_y=True)
            self.obj.set_pose(Pose.create_from_pq(p=xyz, q=qs))
            self._store_initial_obj_z(env_idx, xyz[:, 2])
            self._settle_after_reset(env_idx)
            self._refresh_reset_reference_state(env_idx)

    def _refresh_reset_reference_state(self, env_idx: torch.Tensor) -> None:
        """Refresh lift baseline from the post-settle cube pose."""
        self._store_initial_obj_z(env_idx, self.obj.pose.p[env_idx, 2])

    def evaluate(self) -> dict[str, torch.Tensor]:
        """Compute per-environment success and intermediate metrics."""
        tcp_to_obj_dist = torch.linalg.norm(self.obj.pose.p - self.agent.tcp_pose.p, axis=1)
        obj_to_target_xy = self.obj.pose.p[:, :2] - self.target_site.pose.p[:, :2]
        obj_to_target_dist = torch.linalg.norm(obj_to_target_xy, axis=1)
        cube_near_ground = self.obj.pose.p[:, 2] < (self.cube_half_size + 0.01)
        is_obj_placed = (obj_to_target_dist <= self._robot_cfg["goal_thresh"]) & cube_near_ground
        is_grasped = self.agent.is_grasping(self.obj)
        is_robot_static = self.agent.is_static()

        obj_z = self.obj.pose.p[:, 2]
        lift_height = obj_z - self._initial_obj_z
        success = is_obj_placed & is_robot_static

        return {
            "obj_to_target_dist": obj_to_target_dist,
            "is_obj_placed": is_obj_placed,
            "is_grasped": is_grasped,
            "is_robot_static": is_robot_static,
            "lift_height": lift_height,
            "success": success,
            "tcp_to_obj_dist": tcp_to_obj_dist,
        }

    def _get_obs_extra(self, info: dict) -> dict[str, torch.Tensor]:
        return self._build_obs_extra_from_components(info)

    def _add_component_obs(
        self, obs: dict[str, torch.Tensor], component: object, info: dict
    ) -> None:
        # Semantics mirror so101_nexus_mujoco.pick_and_place.PickAndPlaceEnv
        # ._get_component_data: ObjectPose = cube pose; ObjectOffset =
        # obj_pos - tcp_pos; TargetPosition = target position; TargetOffset =
        # target_pos - obj_pos.
        if isinstance(component, ObjectPose):
            obs["object_pose"] = self.obj.pose.raw_pose
        elif isinstance(component, ObjectOffset):
            obs["object_offset"] = self.obj.pose.p - self.agent.tcp_pose.p
        elif isinstance(component, TargetPosition):
            obs["target_position"] = self.target_site.pose.p
        elif isinstance(component, TargetOffset):
            obs["target_offset"] = self.target_site.pose.p - self.obj.pose.p
        else:
            super()._add_component_obs(obs, component, info)

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict) -> torch.Tensor:
        """Compute the normalized dense reward for pick-and-place."""
        reach_progress = self._reach_progress(info["tcp_to_obj_dist"])
        is_grasped = info["is_grasped"]
        placement_progress = self._reach_progress(info["obj_to_target_dist"]) * is_grasped

        # Norms are stamped once per step in get_reward; read, do not recompute.
        return self._assemble_normalized_reward(
            reach_progress=reach_progress,
            is_grasped=is_grasped,
            task_progress=placement_progress,
            is_complete=info["success"],
            action_delta_norm=info["action_delta_norm"],
            energy_norm=info["energy_norm"],
        )


PickAndPlaceSO100Env = register_robot_variant(
    class_name="PickAndPlaceSO100Env",
    env_id="ManiSkillPickAndPlaceSO100-v1",
    base_cls=PickAndPlaceEnv,
    robot_uid="so100",
    max_episode_steps=1024,
    caller_globals=globals(),
)
PickAndPlaceSO101Env = register_robot_variant(
    class_name="PickAndPlaceSO101Env",
    env_id="ManiSkillPickAndPlaceSO101-v1",
    base_cls=PickAndPlaceEnv,
    robot_uid="so101",
    max_episode_steps=1024,
    caller_globals=globals(),
)
