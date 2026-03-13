from typing import Any

import numpy as np
import sapien
import torch
from mani_skill.envs.utils.randomization.pose import random_quaternions
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose

from so101_nexus_core.config import (
    YCB_OBJECTS,
    PickYCBMultipleConfig,
)
from so101_nexus_core.robot_presets import build_maniskill_robot_configs
from so101_nexus_core.ycb_geometry import get_maniskill_ycb_spawn_z
from so101_nexus_maniskill.base_env import SO101NexusManiSkillBaseEnv

_DEFAULT_CONFIG = PickYCBMultipleConfig()
PICK_YCB_MULTIPLE_CONFIGS: dict[str, dict] = build_maniskill_robot_configs(config=_DEFAULT_CONFIG)


class PickYCBMultipleLiftEnv(SO101NexusManiSkillBaseEnv):
    """Pick the target YCB object from distractors and lift it while grasped."""

    config: PickYCBMultipleConfig

    def __init__(
        self,
        *args,
        config: PickYCBMultipleConfig = PickYCBMultipleConfig(),
        robot_uids: str = "so100",
        num_envs: int = 1,
        reconfiguration_freq: int | None = None,
        **kwargs,
    ):
        rng = np.random.default_rng()
        available = list(config.available_model_ids)
        self.model_id = str(rng.choice(available))
        self.num_distractors = config.num_distractors
        self.min_object_separation = config.min_object_separation
        self._obj_spawn_z = 0.0
        self._distractor_spawn_zs: list[float] = []
        self.task_description = (
            f"Pick up the {YCB_OBJECTS[self.model_id]} from among"
            f" {config.num_distractors} distractors"
        )

        distractor_pool = [mid for mid in available if mid != self.model_id]
        if not distractor_pool:
            distractor_pool = available
        self.distractor_model_ids: list[str] = list(
            rng.choice(distractor_pool, size=config.num_distractors, replace=True)
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

    def _after_reconfigure(self, options: dict) -> None:
        super()._after_reconfigure(options)
        self._obj_spawn_z = get_maniskill_ycb_spawn_z(self.model_id)
        self._distractor_spawn_zs = [
            get_maniskill_ycb_spawn_z(mid) for mid in self.distractor_model_ids
        ]

    def _load_scene(self, options: dict) -> None:
        """Build one merged target actor and one merged actor per distractor slot."""
        self._build_ground()

        objs: list[Actor] = []
        for i in range(self.num_envs):
            builder = actors.get_actor_builder(self.scene, id=f"ycb:{self.model_id}")
            builder.initial_pose = sapien.Pose(p=[0, 0, 0])
            builder.set_scene_idxs([i])
            obj = builder.build(name=f"ycb_target-{i}")
            objs.append(obj)
            self.remove_from_state_dict_registry(obj)
        self.obj = Actor.merge(objs, name="ycb_target")
        self.add_to_state_dict_registry(self.obj)

        self.distractors: list[Actor] = []
        for d in range(self.num_distractors):
            d_objs: list[Actor] = []
            for i in range(self.num_envs):
                builder = actors.get_actor_builder(
                    self.scene, id=f"ycb:{self.distractor_model_ids[d]}"
                )
                builder.initial_pose = sapien.Pose(p=[0, 0, 0])
                builder.set_scene_idxs([i])
                d_obj = builder.build(name=f"ycb_distractor_{d}-{i}")
                d_objs.append(d_obj)
                self.remove_from_state_dict_registry(d_obj)
            distractor = Actor.merge(d_objs, name=f"ycb_distractor_{d}")
            self.add_to_state_dict_registry(distractor)
            self.distractors.append(distractor)

        self._apply_robot_color_if_needed()

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict) -> None:
        """Reset robots and sample separated target and distractor YCB poses."""
        with torch.device(self.device):
            b = len(env_idx)
            self._reset_robot(env_idx)

            cfg = self._robot_cfg
            min_r = cfg["spawn_min_radius"]
            max_r = cfg["spawn_max_radius"]
            angle_half = cfg["spawn_angle_half_range"]
            obj_spawn_z = float(self._obj_spawn_z)

            total_objects = 1 + self.num_distractors

            bounding_radii = [obj_spawn_z]
            for dz in self._distractor_spawn_zs:
                bounding_radii.append(float(dz))

            all_xy = []
            for _ in range(b):
                positions = _sample_separated_positions_polar_np(
                    total_objects,
                    min_r,
                    max_r,
                    angle_half,
                    self.min_object_separation,
                    bounding_radii,
                )
                all_xy.append(positions)

            target_xyz = torch.zeros((b, 3), device=self.device)
            for bi in range(b):
                target_xyz[bi, 0] = all_xy[bi][0][0]
                target_xyz[bi, 1] = all_xy[bi][0][1]
            target_xyz[:, 2] = obj_spawn_z
            qs = random_quaternions(b, lock_x=True, lock_y=True)
            self.obj.set_pose(Pose.create_from_pq(p=target_xyz, q=qs))
            self._store_initial_obj_z(env_idx, target_xyz[:, 2])

            for d in range(self.num_distractors):
                d_xyz = torch.zeros((b, 3), device=self.device)
                for bi in range(b):
                    d_xyz[bi, 0] = all_xy[bi][1 + d][0]
                    d_xyz[bi, 1] = all_xy[bi][1 + d][1]
                d_xyz[:, 2] = self._distractor_spawn_zs[d]
                d_qs = random_quaternions(b, lock_x=True, lock_y=True)
                self.distractors[d].set_pose(Pose.create_from_pq(p=d_xyz, q=d_qs))

    def evaluate(self) -> dict[str, torch.Tensor]:
        tcp_to_obj_dist = torch.linalg.norm(self.obj.pose.p - self.agent.tcp_pose.p, axis=1)
        is_grasped = self.agent.is_grasping(self.obj)

        obj_z = self.obj.pose.p[:, 2]
        lift_height = obj_z - self._initial_obj_z
        success = (lift_height > self.config.lift_threshold) & is_grasped

        return {
            "is_grasped": is_grasped,
            "lift_height": lift_height,
            "success": success,
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
        reach_progress = self._reach_progress(info["tcp_to_obj_dist"])
        is_grasped = info["is_grasped"]
        lift_progress = (
            torch.tanh(self.config.reward.tanh_shaping_scale * info["lift_height"].clamp(min=0.0))
            * is_grasped
        )

        return self._assemble_normalized_reward(
            reach_progress=reach_progress,
            is_grasped=is_grasped,
            task_progress=lift_progress,
            is_complete=info["success"],
        )


def _sample_separated_positions_polar_np(
    count: int,
    min_r: float,
    max_r: float,
    angle_half: float,
    min_clearance: float,
    bounding_radii: list[float],
    max_attempts: int = 100,
) -> list[tuple[float, float]]:
    """Sample 2D positions in a polar arc with bounding-radius-aware separation."""
    positions: list[tuple[float, float]] = []
    for idx in range(count):
        for _ in range(max_attempts):
            r = np.random.uniform(min_r, max_r)
            theta = np.random.uniform(-angle_half, angle_half)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
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


PickYCBMultipleLiftSO100Env = _register_robot_variant(
    class_name="PickYCBMultipleLiftSO100Env",
    env_id="ManiSkillPickYCBMultipleLiftSO100-v1",
    base_cls=PickYCBMultipleLiftEnv,
    robot_uid="so100",
)
PickYCBMultipleLiftSO101Env = _register_robot_variant(
    class_name="PickYCBMultipleLiftSO101Env",
    env_id="ManiSkillPickYCBMultipleLiftSO101-v1",
    base_cls=PickYCBMultipleLiftEnv,
    robot_uid="so101",
)
