"""ManiSkill unified pick environment.

Provides ``PickEnv`` (reach-only reward) and ``PickLiftEnv`` (lift-to-success)
backed by a ManiSkill scene built from a ``PickConfig`` object list.

Supported object types: ``CubeObject``, ``YCBObject``.
``MeshObject`` is not yet supported on the ManiSkill backend.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import sapien
import torch
from mani_skill.envs.utils.randomization.pose import random_quaternions
from mani_skill.utils.building import actors
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose

from so101_nexus_core.config import PickConfig, describe_pick_target
from so101_nexus_core.constants import sample_color
from so101_nexus_core.objects import CubeObject, SceneObject, YCBObject
from so101_nexus_core.observations import ObjectOffset, ObjectPose
from so101_nexus_core.robot_presets import build_maniskill_robot_configs
from so101_nexus_core.ycb_geometry import (
    get_maniskill_ycb_bounding_radius,
    get_maniskill_ycb_spawn_z,
)
from so101_nexus_maniskill.base_env import SO101NexusManiSkillBaseEnv, register_robot_variant
from so101_nexus_maniskill.spawn_utils import sample_separated_positions_torch

# Default bounding radius used when an object type has no computable extent.
_DEFAULT_BOUNDING_RADIUS = 0.025


def _pick_target_and_distractors(
    rng: np.random.Generator | np.random.RandomState,
    objects: list[SceneObject],
    n_distractors: int,
) -> tuple[SceneObject, list[SceneObject]]:
    """Randomly select a target and distractor objects from the pool.

    Distractors are drawn without replacement from the non-target pool, matching
    the MuJoCo backend (``rng.choice(..., replace=False)``), so no distractor
    identity is duplicated when the pool is large enough.

    Divergence from the MuJoCo backend: this backend samples the target
    uniformly over the full pool, then distractors uniformly over the remainder,
    and tolerates ``n_active > pool`` via a ``replace=True`` fallback. MuJoCo
    instead samples target plus distractors jointly without replacement and
    would raise on such overflow. Both guarantee distractor uniqueness when the
    pool allows.

    Parameters
    ----------
    rng:
        NumPy random generator or legacy ``RandomState`` (ManiSkill's seeded
        episode RNG is a ``RandomState``); both expose ``choice``.
    objects:
        Pool of scene objects to sample from.
    n_distractors:
        Number of distractor objects to select.

    Returns
    -------
    target, distractors:
        The chosen target and a list of distractor objects.
    """
    n_pool = len(objects)
    target_idx = int(rng.choice(n_pool))
    target = objects[target_idx]

    distractor_pool_idx = [i for i in range(n_pool) if i != target_idx]
    if not distractor_pool_idx:
        # Single-object pool: the only object doubles as a distractor.
        distractors = [objects[target_idx] for _ in range(n_distractors)]
        return target, distractors

    # Without replacement when the pool allows; if more distractors than the
    # pool can supply uniquely, fall back to sampling with replacement.
    replace = n_distractors > len(distractor_pool_idx)
    chosen = rng.choice(distractor_pool_idx, size=n_distractors, replace=replace)
    distractors = [objects[int(i)] for i in chosen]
    return target, distractors


def _obj_bounding_radius(obj: SceneObject) -> float:
    """Return the XY bounding radius for separation-aware placement, in metres."""
    if isinstance(obj, CubeObject):
        return float(obj.half_size * np.sqrt(2))
    if isinstance(obj, YCBObject):
        return get_maniskill_ycb_bounding_radius(obj.model_id)
    return _DEFAULT_BOUNDING_RADIUS


def _obj_spawn_z(obj: SceneObject) -> float:
    """Return the spawn height (z) for the bottom of the object to rest on the floor."""
    if isinstance(obj, CubeObject):
        return obj.half_size
    if isinstance(obj, YCBObject):
        return get_maniskill_ycb_spawn_z(obj.model_id)
    raise TypeError(f"Unsupported object type for spawn_z: {type(obj)}")


def _build_actor(
    env: SO101NexusManiSkillBaseEnv,
    obj: SceneObject,
    name: str,
    env_idx: int,
) -> Actor:
    """Build a single actor in the ManiSkill scene for one environment index.

    Parameters
    ----------
    env:
        The ManiSkill environment owning the scene.
    obj:
        The scene object specification.
    name:
        Actor name (must be unique within the merged group).
    env_idx:
        The environment index this actor belongs to.

    Returns
    -------
    Actor
        The built (unmerged) actor.
    """
    if isinstance(obj, CubeObject):
        return actors.build_cube(
            env.scene,
            half_size=obj.half_size,
            color=sample_color(obj.color),
            name=name,
            body_type="dynamic",
            scene_idxs=[env_idx],
            initial_pose=sapien.Pose(p=[0, 0, obj.half_size], q=[1, 0, 0, 0]),
        )
    if isinstance(obj, YCBObject):
        builder = actors.get_actor_builder(env.scene, id=f"ycb:{obj.model_id}")
        builder.initial_pose = sapien.Pose(p=[0, 0, _obj_spawn_z(obj)], q=[1, 0, 0, 0])
        builder.set_scene_idxs([env_idx])
        return builder.build(name=name)
    raise TypeError(
        f"Unsupported object type on ManiSkill backend: {type(obj).__name__}. "
        "Only CubeObject and YCBObject are supported."
    )


class PickEnv(SO101NexusManiSkillBaseEnv):
    """Unified ManiSkill pick environment with reach-only reward.

    Handles ``CubeObject`` and ``YCBObject`` from ``PickConfig.objects``.
    One object is randomly chosen as the target at each scene load; distractors
    fill remaining slots. Task description is auto-generated from
    ``repr(target_obj)``.

    Parameters
    ----------
    config:
        Environment configuration. Defaults to ``PickConfig()``.
    robot_uids:
        Robot identifier: ``"so100"`` or ``"so101"``.
    num_envs:
        Number of parallel environments.
    reconfiguration_freq:
        How often to rebuild the scene. ``None`` uses the default heuristic
        (rebuild each episode when wrist camera is enabled).
    """

    config: PickConfig
    default_config_cls: ClassVar[type[PickConfig]] = PickConfig

    def __init__(
        self,
        *args,
        config: PickConfig | None = None,
        robot_uids: str = "so100",
        num_envs: int = 1,
        reconfiguration_freq: int | None = None,
        robot_init_qpos_noise: float | None = None,
        **kwargs,
    ):
        if config is None:
            config = PickConfig()
        robot_cfgs = build_maniskill_robot_configs(config=config)

        self._setup_base(
            config=config,
            robot_uids=robot_uids,
            robot_cfgs=robot_cfgs,
            robot_init_qpos_noise=robot_init_qpos_noise,
        )

        self._target_obj: SceneObject | None = None
        self._distractors_spec: list[SceneObject] = []
        self._task_description: str = ""
        self._obj_spawn_z_val: float = 0.0
        self._distractor_spawn_zs: list[float] = []
        # Bounding radii (target first, then distractors) for separation-aware
        # placement; populated in _load_scene.
        self._spawn_bounding_radii: list[float] = []

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

    @property
    def task_description(self) -> str:
        """Current episode task description derived from target object repr."""
        return self._task_description

    def _default_reconfiguration_freq(self) -> int:
        """Reconfigure each episode when the object pool can vary the identities.

        Identity selection (target and distractors) happens at scene load, so a
        variable-object Pick scene (pool larger than the active target+distractor
        count) must reconfigure every episode for identities to actually change.
        Single-object or fixed scenes keep the camera-driven base heuristic
        (0 unless a wrist camera forces a per-episode rebuild) to avoid needless
        scene rebuilds.
        """
        n_pool = len(self.config.objects)
        n_active = 1 + self.config.n_distractors
        if n_pool > n_active:
            return 1
        return super()._default_reconfiguration_freq()

    def _load_scene(self, options: dict) -> None:
        self._build_ground()

        # Sample target and distractor IDENTITIES with the seeded episode RNG so
        # repeated reset(seed=S) reproduces the same identities. ManiSkill sets
        # self._episode_rng (a seeded np.random.RandomState) before _reconfigure
        # / _load_scene runs (see mani_skill.envs.sapien_env.BaseEnv.reset).
        # Fall back to the global RNG only if accessed before the first seeded
        # reset (e.g. during construction's initial reconfigure).
        rng: np.random.Generator | np.random.RandomState = (
            self._episode_rng
            if getattr(self, "_episode_rng", None) is not None
            else np.random.default_rng()
        )
        target, distractors = _pick_target_and_distractors(
            rng,
            self.config.objects,
            self.config.n_distractors,
        )
        self._target_obj = target
        self._distractors_spec = distractors
        self._task_description = describe_pick_target(target)
        self._obj_spawn_z_val = _obj_spawn_z(target)
        self._distractor_spawn_zs = [_obj_spawn_z(d) for d in distractors]
        self._spawn_bounding_radii = [_obj_bounding_radius(target)] + [
            _obj_bounding_radius(d) for d in distractors
        ]

        # Build vectorized target actor (one per env, merged into one).
        target_actors: list[Actor] = []
        for i in range(self.num_envs):
            actor = _build_actor(self, target, f"pick_target-{i}", i)
            target_actors.append(actor)
            self.remove_from_state_dict_registry(actor)
        self.obj = Actor.merge(target_actors, name="pick_target")
        self.add_to_state_dict_registry(self.obj)

        # Build vectorized distractor actors.
        self.distractors: list[Actor] = []
        for d_idx, d_obj in enumerate(distractors):
            d_actors: list[Actor] = []
            for i in range(self.num_envs):
                actor = _build_actor(self, d_obj, f"distractor_{d_idx}-{i}", i)
                d_actors.append(actor)
                self.remove_from_state_dict_registry(actor)
            distractor = Actor.merge(d_actors, name=f"distractor_{d_idx}")
            self.add_to_state_dict_registry(distractor)
            self.distractors.append(distractor)

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

            # Sample target + distractor XY with bounding-radius-aware separation,
            # enforcing config.min_object_separation. Mirrors the MuJoCo backend's
            # sample_separated_positions (so101_nexus_mujoco.spawn_utils) but
            # batched per env row. positions: (b, total_objects, 2).
            bounding_radii = torch.tensor(
                self._spawn_bounding_radii, dtype=torch.float32, device=self.device
            )
            positions = sample_separated_positions_torch(
                num_envs=b,
                bounding_radii=bounding_radii,
                min_r=min_r,
                max_r=max_r,
                angle_half=angle_half,
                min_clearance=self.config.min_object_separation,
                center=(cx, cy),
                device=self.device,
            )

            # Target pose
            target_xyz = torch.zeros((b, 3), device=self.device)
            target_xyz[:, :2] = positions[:, 0, :]
            target_xyz[:, 2] = self._obj_spawn_z_val
            qs = random_quaternions(b, lock_x=True, lock_y=True)
            self.obj.set_pose(Pose.create_from_pq(p=target_xyz, q=qs))

            # Distractor poses
            for d_idx, distractor in enumerate(self.distractors):
                slot = 1 + d_idx
                d_xyz = torch.zeros((b, 3), device=self.device)
                d_xyz[:, :2] = positions[:, slot, :]
                d_xyz[:, 2] = self._distractor_spawn_zs[d_idx]
                d_qs = random_quaternions(b, lock_x=True, lock_y=True)
                distractor.set_pose(Pose.create_from_pq(p=d_xyz, q=d_qs))

            self._settle_after_reset(env_idx)
            self._refresh_reset_reference_state(env_idx)

    def _refresh_reset_reference_state(self, env_idx: torch.Tensor) -> None:
        """Refresh lift baseline from the post-settle target object pose."""
        self._store_initial_obj_z(env_idx, self.obj.pose.p[env_idx, 2])

    def evaluate(self) -> dict[str, torch.Tensor]:
        """Return per-env metrics: is_grasped, lift_height, tcp_to_obj_dist."""
        tcp_to_obj_dist = torch.linalg.norm(self.obj.pose.p - self.agent.tcp_pose.p, axis=1)
        is_grasped = self.agent.is_grasping(self.obj)

        obj_z = self.obj.pose.p[:, 2]
        lift_height = obj_z - self._initial_obj_z

        return {
            "is_grasped": is_grasped,
            "lift_height": lift_height,
            "tcp_to_obj_dist": tcp_to_obj_dist,
        }

    def _get_obs_extra(self, info: dict) -> dict[str, torch.Tensor]:
        return self._build_obs_extra_from_components(info)

    def _add_component_obs(
        self, obs: dict[str, torch.Tensor], component: object, info: dict
    ) -> None:
        # Semantics mirror so101_nexus_mujoco.pick_env.PickEnv._get_component_data:
        # ObjectPose = object pose; ObjectOffset = obj_pos - tcp_pos.
        if isinstance(component, ObjectPose):
            obs["object_pose"] = self.obj.pose.raw_pose
        elif isinstance(component, ObjectOffset):
            obs["object_offset"] = self.obj.pose.p - self.agent.tcp_pose.p
        else:
            super()._add_component_obs(obs, component, info)

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict) -> torch.Tensor:
        """Return reach-only reward normalized to [0, 1]."""
        reach_progress = self._reach_progress(info["tcp_to_obj_dist"])
        is_grasped = info["is_grasped"]

        # Norms are stamped once per step in get_reward; read, do not recompute.
        return self._assemble_normalized_reward(
            reach_progress=reach_progress,
            is_grasped=is_grasped,
            task_progress=torch.zeros_like(reach_progress),
            is_complete=info.get(
                "success",
                torch.zeros(len(reach_progress), dtype=torch.bool, device=self.device),
            ),
            action_delta_norm=info["action_delta_norm"],
            energy_norm=info["energy_norm"],
        )


class PickLiftEnv(PickEnv):
    """Pick-lift variant where success requires grasping and lifting the target.

    Extends ``PickEnv`` with a lift reward and a ``success`` flag in the info
    dict. Success condition: ``lift_height > config.lift_threshold`` while grasped.
    """

    def evaluate(self) -> dict[str, torch.Tensor]:
        """Return per-env metrics including success flag based on lift threshold."""
        info = super().evaluate()
        info["success"] = (info["lift_height"] > self.config.lift_threshold) & info["is_grasped"]
        return info

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict) -> torch.Tensor:
        """Return lift reward normalized to [0, 1]."""
        reach_progress = self._reach_progress(info["tcp_to_obj_dist"])
        is_grasped = info["is_grasped"]
        lift_progress = (
            torch.tanh(self.config.reward.tanh_shaping_scale * info["lift_height"].clamp(min=0.0))
            * is_grasped
        )

        # Norms are stamped once per step in get_reward; read, do not recompute.
        return self._assemble_normalized_reward(
            reach_progress=reach_progress,
            is_grasped=is_grasped,
            task_progress=lift_progress,
            is_complete=info["success"],
            action_delta_norm=info["action_delta_norm"],
            energy_norm=info["energy_norm"],
        )


PickLiftSO100Env = register_robot_variant(
    class_name="PickLiftSO100Env",
    env_id="ManiSkillPickLiftSO100-v1",
    base_cls=PickLiftEnv,
    robot_uid="so100",
    max_episode_steps=1024,
    caller_globals=globals(),
)
PickLiftSO101Env = register_robot_variant(
    class_name="PickLiftSO101Env",
    env_id="ManiSkillPickLiftSO101-v1",
    base_cls=PickLiftEnv,
    robot_uid="so101",
    max_episode_steps=1024,
    caller_globals=globals(),
)
