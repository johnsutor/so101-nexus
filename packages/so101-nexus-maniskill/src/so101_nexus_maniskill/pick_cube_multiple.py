from typing import Any

import numpy as np
import torch
from mani_skill.envs.utils.randomization.pose import random_quaternions
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose

from so101_nexus_core.config import (
    CUBE_COLOR_MAP,
    PickCubeMultipleConfig,
)
from so101_nexus_core.robot_presets import build_maniskill_robot_configs
from so101_nexus_maniskill.base_env import SO101NexusManiSkillBaseEnv

_DEFAULT_CONFIG = PickCubeMultipleConfig()
PICK_CUBE_MULTIPLE_CONFIGS: dict[str, dict] = build_maniskill_robot_configs(config=_DEFAULT_CONFIG)


@register_env(
    "ManiSkillPickCubeMultipleGoal-v1",
    max_episode_steps=_DEFAULT_CONFIG.max_episode_steps,
)
class PickCubeMultipleEnv(SO101NexusManiSkillBaseEnv):
    """Configurable pick-cube-multiple environment supporting SO100 and SO101 robots."""

    config: PickCubeMultipleConfig

    def __init__(
        self,
        *args,
        config: PickCubeMultipleConfig = PickCubeMultipleConfig(),
        robot_uids: str = "so100",
        num_envs: int = 1,
        reconfiguration_freq: int | None = None,
        **kwargs,
    ):
        self.cube_color_name = config.cube_color
        self.cube_color_rgba = CUBE_COLOR_MAP[config.cube_color]
        self.cube_half_size = config.cube_half_size
        self.num_distractors = config.num_distractors
        self.min_object_separation = config.min_object_separation
        self.task_description = (
            f"Pick up the small {config.cube_color} cube"
            f" from among {config.num_distractors} distractors"
        )

        robot_cfgs = build_maniskill_robot_configs(config=config)

        self._setup_base(
            config=config,
            robot_uids=robot_uids,
            robot_cfgs=robot_cfgs,
        )

        if reconfiguration_freq is None:
            reconfiguration_freq = 1 if config.camera_mode in ("wrist", "both") else 0

        super().__init__(
            *args,
            robot_uids=robot_uids,
            reconfiguration_freq=reconfiguration_freq,
            num_envs=num_envs,
            **kwargs,
        )

    def _load_scene(self, options: dict) -> None:
        self._build_ground()

        # Build target cubes (one per env, merged)
        objs: list[Actor] = []
        for i in range(self.num_envs):
            cube = actors.build_cube(
                self.scene,
                half_size=self.cube_half_size,
                color=self.cube_color_rgba,
                name=f"cube_target-{i}",
                body_type="dynamic",
                scene_idxs=[i],
            )
            objs.append(cube)
            self.remove_from_state_dict_registry(cube)
        self.obj = Actor.merge(objs, name="cube_target")
        self.add_to_state_dict_registry(self.obj)

        # Build distractor cubes
        distractor_color_names = [c for c in CUBE_COLOR_MAP if c != self.cube_color_name]
        self.distractors: list[Actor] = []
        for d in range(self.num_distractors):
            d_objs: list[Actor] = []
            for i in range(self.num_envs):
                color_name = distractor_color_names[np.random.randint(len(distractor_color_names))]
                cube = actors.build_cube(
                    self.scene,
                    half_size=self.cube_half_size,
                    color=CUBE_COLOR_MAP[color_name],
                    name=f"cube_distractor_{d}-{i}",
                    body_type="dynamic",
                    scene_idxs=[i],
                )
                d_objs.append(cube)
                self.remove_from_state_dict_registry(cube)
            distractor = Actor.merge(d_objs, name=f"cube_distractor_{d}")
            self.add_to_state_dict_registry(distractor)
            self.distractors.append(distractor)

        # Goal sphere
        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self._robot_cfg["goal_thresh"],
            color=[0, 1, 0, 0.5],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
        )
        self._hidden_objects.append(self.goal_site)
        self._apply_robot_color_if_needed()

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict) -> None:
        with torch.device(self.device):
            b = len(env_idx)
            self._reset_robot(env_idx)

            cfg = self._robot_cfg
            spawn_cx, spawn_cy = cfg["cube_spawn_center"]
            spawn_hs = cfg["cube_spawn_half_size"]

            total_objects = 1 + self.num_distractors
            cube_radius = self.cube_half_size * np.sqrt(2)

            # Sample separated positions for all objects per env
            all_xyz = torch.zeros((b, total_objects, 3), device=self.device)
            for bi in range(b):
                positions = _sample_separated_positions_np(
                    total_objects,
                    spawn_cx,
                    spawn_cy,
                    spawn_hs,
                    self.min_object_separation,
                    [cube_radius] * total_objects,
                )
                for oi, (px, py) in enumerate(positions):
                    all_xyz[bi, oi, 0] = px
                    all_xyz[bi, oi, 1] = py
                    all_xyz[bi, oi, 2] = self.cube_half_size

            # Set target pose
            target_xyz = all_xyz[:, 0]
            qs = random_quaternions(b, lock_x=True, lock_y=True)
            self.obj.set_pose(Pose.create_from_pq(p=target_xyz, q=qs))
            self._store_initial_obj_z(env_idx, target_xyz[:, 2])

            # Set distractor poses
            for d in range(self.num_distractors):
                d_xyz = all_xyz[:, 1 + d]
                d_qs = random_quaternions(b, lock_x=True, lock_y=True)
                self.distractors[d].set_pose(Pose.create_from_pq(p=d_xyz, q=d_qs))

            # Set goal pose
            goal_xyz = torch.zeros((b, 3), device=self.device)
            goal_xyz[:, 0] = spawn_cx + (torch.rand(b, device=self.device) * 2 - 1) * spawn_hs
            goal_xyz[:, 1] = spawn_cy + (torch.rand(b, device=self.device) * 2 - 1) * spawn_hs
            goal_xyz[:, 2] = (
                self.cube_half_size + torch.rand(b, device=self.device) * cfg["max_goal_height"]
            )
            self.goal_site.set_pose(Pose.create_from_pq(p=goal_xyz))

    def evaluate(self) -> dict[str, torch.Tensor]:
        tcp_to_obj_dist = torch.linalg.norm(self.obj.pose.p - self.agent.tcp_pose.p, axis=1)
        obj_to_goal_dist = torch.linalg.norm(self.obj.pose.p - self.goal_site.pose.p, axis=1)
        is_obj_placed = obj_to_goal_dist <= self._robot_cfg["goal_thresh"]
        is_grasped = self.agent.is_grasping(self.obj)
        is_robot_static = self.agent.is_static()

        obj_z = self.obj.pose.p[:, 2]
        lift_height = obj_z - self._initial_obj_z
        success = is_obj_placed & is_robot_static

        return {
            "obj_to_goal_dist": obj_to_goal_dist,
            "is_obj_placed": is_obj_placed,
            "is_grasped": is_grasped,
            "is_robot_static": is_robot_static,
            "lift_height": lift_height,
            "success": success,
            "tcp_to_obj_dist": tcp_to_obj_dist,
        }

    def _get_obs_extra(self, info: dict) -> dict[str, torch.Tensor]:
        obs = {
            "tcp_pose": self.agent.tcp_pose.raw_pose,
            "is_grasped": info["is_grasped"],
            "goal_pos": self.goal_site.pose.p,
        }
        if "state" in self.obs_mode:
            obs.update(
                {
                    "obj_pose": self.obj.pose.raw_pose,
                    "tcp_to_obj_pos": self.obj.pose.p - self.agent.tcp_pose.p,
                    "obj_to_goal_pos": self.goal_site.pose.p - self.obj.pose.p,
                }
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict) -> torch.Tensor:
        reach_progress = self._reach_progress(info["tcp_to_obj_dist"])
        is_grasped = info["is_grasped"]
        placement_progress = self._reach_progress(info["obj_to_goal_dist"]) * is_grasped

        return self._assemble_normalized_reward(
            reach_progress=reach_progress,
            is_grasped=is_grasped,
            task_progress=placement_progress,
            is_complete=info["success"],
        )


PickCubeMultipleGoalEnv = PickCubeMultipleEnv


@register_env(
    "ManiSkillPickCubeMultipleLift-v1",
    max_episode_steps=_DEFAULT_CONFIG.max_episode_steps,
)
class PickCubeMultipleLiftEnv(PickCubeMultipleEnv):
    """Pick-cube-multiple variant where success is lift height threshold while grasped."""

    def evaluate(self) -> dict[str, torch.Tensor]:
        info = super().evaluate()
        info["success"] = (info["lift_height"] > self.config.lift_threshold) & info["is_grasped"]
        return info

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict) -> torch.Tensor:
        reach_progress = self._reach_progress(info["tcp_to_obj_dist"])
        is_grasped = info["is_grasped"]
        lift_progress = torch.tanh(5.0 * info["lift_height"].clamp(min=0.0)) * is_grasped

        return self._assemble_normalized_reward(
            reach_progress=reach_progress,
            is_grasped=is_grasped,
            task_progress=lift_progress,
            is_complete=info["success"],
        )


def _sample_separated_positions_np(
    count: int,
    cx: float,
    cy: float,
    spawn_hs: float,
    min_clearance: float,
    bounding_radii: list[float],
    max_attempts: int = 100,
) -> list[tuple[float, float]]:
    """Sample 2D positions with bounding-radius-aware separation (numpy RNG)."""
    positions: list[tuple[float, float]] = []
    for idx in range(count):
        for _ in range(max_attempts):
            x = cx + np.random.uniform(-spawn_hs, spawn_hs)
            y = cy + np.random.uniform(-spawn_hs, spawn_hs)
            if all(
                np.sqrt((x - px) ** 2 + (y - py) ** 2)
                >= bounding_radii[idx] + bounding_radii[j] + min_clearance
                for j, (px, py) in enumerate(positions)
            ):
                positions.append((x, y))
                break
        else:
            positions.append((x, y))
    return positions


def _register_robot_variant(
    *,
    class_name: str,
    env_id: str,
    base_cls: type,
    robot_uid: str,
) -> type:
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("robot_uids", robot_uid)
        base_cls.__init__(self, *args, **kwargs)

    cls = type(class_name, (base_cls,), {"__init__": __init__})
    cls = register_env(env_id, max_episode_steps=_DEFAULT_CONFIG.max_episode_steps)(cls)
    globals()[class_name] = cls
    return cls


PickCubeMultipleGoalSO100Env = _register_robot_variant(
    class_name="PickCubeMultipleGoalSO100Env",
    env_id="ManiSkillPickCubeMultipleGoalSO100-v1",
    base_cls=PickCubeMultipleEnv,
    robot_uid="so100",
)
PickCubeMultipleGoalSO101Env = _register_robot_variant(
    class_name="PickCubeMultipleGoalSO101Env",
    env_id="ManiSkillPickCubeMultipleGoalSO101-v1",
    base_cls=PickCubeMultipleEnv,
    robot_uid="so101",
)
PickCubeMultipleLiftSO100Env = _register_robot_variant(
    class_name="PickCubeMultipleLiftSO100Env",
    env_id="ManiSkillPickCubeMultipleLiftSO100-v1",
    base_cls=PickCubeMultipleLiftEnv,
    robot_uid="so100",
)
PickCubeMultipleLiftSO101Env = _register_robot_variant(
    class_name="PickCubeMultipleLiftSO101Env",
    env_id="ManiSkillPickCubeMultipleLiftSO101-v1",
    base_cls=PickCubeMultipleLiftEnv,
    robot_uid="so101",
)
