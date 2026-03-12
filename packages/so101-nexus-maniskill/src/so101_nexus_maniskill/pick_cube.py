from typing import Any

import torch
from mani_skill.envs.utils.randomization.pose import random_quaternions
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose

from so101_nexus_core.config import (
    PickCubeConfig,
    sample_color,
)
from so101_nexus_core.robot_presets import build_maniskill_robot_configs
from so101_nexus_maniskill.base_env import SO101NexusManiSkillBaseEnv

_DEFAULT_CONFIG = PickCubeConfig()
PICK_CUBE_CONFIGS: dict[str, dict] = build_maniskill_robot_configs(config=_DEFAULT_CONFIG)


class PickCubeEnv(SO101NexusManiSkillBaseEnv):
    """Configurable pick-cube base environment supporting SO100 and SO101 robots."""

    config: PickCubeConfig

    def __init__(
        self,
        *args,
        config: PickCubeConfig = PickCubeConfig(),
        robot_uids: str = "so100",
        num_envs: int = 1,
        reconfiguration_freq: int | None = None,
        **kwargs,
    ):
        self.cube_colors = config.cube_colors
        self.cube_half_size = config.cube_half_size
        cube_color_name = (
            config.cube_colors if isinstance(config.cube_colors, str) else config.cube_colors[0]
        )
        self.task_description = f"Pick up the small {cube_color_name} cube"

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

        objs: list[Actor] = []
        for i in range(self.num_envs):
            cube = actors.build_cube(
                self.scene,
                half_size=self.cube_half_size,
                color=sample_color(self.cube_colors),
                name=f"cube-{i}",
                body_type="dynamic",
                scene_idxs=[i],
            )
            objs.append(cube)
            self.remove_from_state_dict_registry(cube)
        self.obj = Actor.merge(objs, name="cube")
        self.add_to_state_dict_registry(self.obj)
        self._apply_robot_color_if_needed()

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict) -> None:
        with torch.device(self.device):
            b = len(env_idx)
            self._reset_robot(env_idx)

            cfg = self._robot_cfg
            min_r = cfg["spawn_min_radius"]
            max_r = cfg["spawn_max_radius"]
            angle_half = cfg["spawn_angle_half_range"]

            r = min_r + torch.rand(b, device=self.device) * (max_r - min_r)
            theta = (torch.rand(b, device=self.device) * 2 - 1) * angle_half

            xyz = torch.zeros((b, 3), device=self.device)
            xyz[:, 0] = r * torch.cos(theta)
            xyz[:, 1] = r * torch.sin(theta)
            xyz[:, 2] = self.cube_half_size
            qs = random_quaternions(b, lock_x=True, lock_y=True)
            self.obj.set_pose(Pose.create_from_pq(p=xyz, q=qs))
            self._store_initial_obj_z(env_idx, xyz[:, 2])

    def evaluate(self) -> dict[str, torch.Tensor]:
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
        reach_progress = self._reach_progress(info["tcp_to_obj_dist"])
        is_grasped = info["is_grasped"]

        return self._assemble_normalized_reward(
            reach_progress=reach_progress,
            is_grasped=is_grasped,
            task_progress=torch.zeros_like(reach_progress),
            is_complete=info.get(
                "success",
                torch.zeros(len(reach_progress), dtype=torch.bool, device=self.device),
            ),
        )


@register_env("ManiSkillPickCubeLift-v1", max_episode_steps=_DEFAULT_CONFIG.max_episode_steps)
class PickCubeLiftEnv(PickCubeEnv):
    """Pick-cube variant where success is lift height threshold while grasped."""

    def evaluate(self) -> dict[str, torch.Tensor]:
        info = super().evaluate()
        info["success"] = (info["lift_height"] > self.config.lift_threshold) & info["is_grasped"]
        return info

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict) -> torch.Tensor:
        reach_progress = self._reach_progress(info["tcp_to_obj_dist"])
        is_grasped = info["is_grasped"]
        lift_progress = torch.tanh(self.config.reward.tanh_shaping_scale * info["lift_height"].clamp(min=0.0)) * is_grasped

        return self._assemble_normalized_reward(
            reach_progress=reach_progress,
            is_grasped=is_grasped,
            task_progress=lift_progress,
            is_complete=info["success"],
        )


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


PickCubeLiftSO100Env = _register_robot_variant(
    class_name="PickCubeLiftSO100Env",
    env_id="ManiSkillPickCubeLiftSO100-v1",
    base_cls=PickCubeLiftEnv,
    robot_uid="so100",
)
PickCubeLiftSO101Env = _register_robot_variant(
    class_name="PickCubeLiftSO101Env",
    env_id="ManiSkillPickCubeLiftSO101-v1",
    base_cls=PickCubeLiftEnv,
    robot_uid="so101",
)
