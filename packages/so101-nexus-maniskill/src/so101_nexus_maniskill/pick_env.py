"""ManiSkill unified pick environment.

Provides ``PickEnv`` (reach-only reward) and ``PickLiftEnv`` (lift-to-success)
backed by a ManiSkill scene built from a ``PickConfig`` object list.

Supported object types: ``CubeObject``, ``YCBObject``.
``MeshObject`` is not yet supported on the ManiSkill backend.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import sapien
import torch
from mani_skill.envs.utils.randomization.pose import random_quaternions
from mani_skill.utils.building import actors
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose

from so101_nexus_core.config import PickConfig
from so101_nexus_core.constants import sample_color
from so101_nexus_core.objects import CubeObject, SceneObject, YCBObject
from so101_nexus_core.robot_presets import build_maniskill_robot_configs
from so101_nexus_core.ycb_geometry import get_maniskill_ycb_spawn_z
from so101_nexus_maniskill.base_env import SO101NexusManiSkillBaseEnv, register_robot_variant

_DEFAULT_CONFIG = PickConfig()


def _pick_target_and_distractors(
    rng: np.random.Generator,
    objects: list[SceneObject],
    n_distractors: int,
) -> tuple[SceneObject, list[SceneObject]]:
    """Randomly select a target and distractor objects from the pool.

    Parameters
    ----------
    rng:
        NumPy random generator.
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
    target_idx = int(rng.integers(n_pool))
    target = objects[target_idx]

    distractor_pool = [o for i, o in enumerate(objects) if i != target_idx]
    if not distractor_pool:
        distractor_pool = list(objects)

    distractors: list[SceneObject] = []
    for _ in range(n_distractors):
        d_idx = int(rng.integers(len(distractor_pool)))
        distractors.append(distractor_pool[d_idx])

    return target, distractors


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
        )
    if isinstance(obj, YCBObject):
        builder = actors.get_actor_builder(env.scene, id=f"ycb:{obj.model_id}")
        builder.initial_pose = sapien.Pose(p=[0, 0, 0])
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

    def __init__(
        self,
        *args,
        config: PickConfig = PickConfig(),
        robot_uids: str = "so100",
        num_envs: int = 1,
        reconfiguration_freq: int | None = None,
        **kwargs,
    ):
        robot_cfgs = build_maniskill_robot_configs(config=config)

        self._setup_base(
            config=config,
            robot_uids=robot_uids,
            robot_cfgs=robot_cfgs,
        )

        self._target_obj: SceneObject | None = None
        self._distractors_spec: list[SceneObject] = []
        self._task_description: str = ""
        self._obj_spawn_z_val: float = 0.0
        self._distractor_spawn_zs: list[float] = []
        self._rng = np.random.default_rng()

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

    def _load_scene(self, options: dict) -> None:
        self._build_ground()

        # Sample target and distractors for this scene configuration.
        target, distractors = _pick_target_and_distractors(
            self._rng,
            self.config.objects,
            self.config.n_distractors,
        )
        self._target_obj = target
        self._distractors_spec = distractors
        self._task_description = f"Pick up the {repr(target)}."
        self._obj_spawn_z_val = _obj_spawn_z(target)
        self._distractor_spawn_zs = [_obj_spawn_z(d) for d in distractors]

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
            self._reset_robot(env_idx)

            cfg = self._robot_cfg
            min_r = cfg["spawn_min_radius"]
            max_r = cfg["spawn_max_radius"]
            angle_half = cfg["spawn_angle_half_range"]

            total_objects = 1 + len(self.distractors)
            all_r = min_r + torch.rand(b, total_objects, device=self.device) * (max_r - min_r)
            all_theta = (torch.rand(b, total_objects, device=self.device) * 2 - 1) * angle_half

            # Target pose
            target_xyz = torch.zeros((b, 3), device=self.device)
            target_xyz[:, 0] = all_r[:, 0] * torch.cos(all_theta[:, 0])
            target_xyz[:, 1] = all_r[:, 0] * torch.sin(all_theta[:, 0])
            target_xyz[:, 2] = self._obj_spawn_z_val
            qs = random_quaternions(b, lock_x=True, lock_y=True)
            self.obj.set_pose(Pose.create_from_pq(p=target_xyz, q=qs))
            self._store_initial_obj_z(env_idx, target_xyz[:, 2])

            # Distractor poses
            for d_idx, distractor in enumerate(self.distractors):
                slot = 1 + d_idx
                d_xyz = torch.zeros((b, 3), device=self.device)
                d_xyz[:, 0] = all_r[:, slot] * torch.cos(all_theta[:, slot])
                d_xyz[:, 1] = all_r[:, slot] * torch.sin(all_theta[:, slot])
                d_xyz[:, 2] = self._distractor_spawn_zs[d_idx]
                d_qs = random_quaternions(b, lock_x=True, lock_y=True)
                distractor.set_pose(Pose.create_from_pq(p=d_xyz, q=d_qs))

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
        obs = {
            "tcp_pose": self.agent.tcp_pose.raw_pose,
            "is_grasped": info["is_grasped"],
        }
        if "state" in self.obs_mode:
            obs.update(
                {
                    "obj_pose": self.obj.pose.raw_pose,
                    "tcp_to_obj_pos": self.obj.pose.p - self.agent.tcp_pose.p,
                }
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict) -> torch.Tensor:
        """Return reach-only reward normalized to [0, 1]."""
        reach_progress = self._reach_progress(info["tcp_to_obj_dist"])
        is_grasped = info["is_grasped"]
        energy_norm = torch.linalg.norm(action, dim=-1)

        return self._assemble_normalized_reward(
            reach_progress=reach_progress,
            is_grasped=is_grasped,
            task_progress=torch.zeros_like(reach_progress),
            is_complete=info.get(
                "success",
                torch.zeros(len(reach_progress), dtype=torch.bool, device=self.device),
            ),
            energy_norm=energy_norm,
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
        energy_norm = torch.linalg.norm(action, dim=-1)

        return self._assemble_normalized_reward(
            reach_progress=reach_progress,
            is_grasped=is_grasped,
            task_progress=lift_progress,
            is_complete=info["success"],
            energy_norm=energy_norm,
        )


PickLiftSO100Env = register_robot_variant(
    class_name="PickLiftSO100Env",
    env_id="ManiSkillPickLiftSO100-v1",
    base_cls=PickLiftEnv,
    robot_uid="so100",
    max_episode_steps=_DEFAULT_CONFIG.max_episode_steps,
    caller_globals=globals(),
)
PickLiftSO101Env = register_robot_variant(
    class_name="PickLiftSO101Env",
    env_id="ManiSkillPickLiftSO101-v1",
    base_cls=PickLiftEnv,
    robot_uid="so101",
    max_episode_steps=_DEFAULT_CONFIG.max_episode_steps,
    caller_globals=globals(),
)
