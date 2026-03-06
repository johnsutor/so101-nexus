from typing import Any

import sapien
import torch
from mani_skill.envs.utils.randomization.pose import random_quaternions
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose

from so101_nexus_core.config import (
    CUBE_COLOR_MAP,
    TARGET_COLOR_MAP,
    PickAndPlaceConfig,
)
from so101_nexus_core.robot_presets import build_maniskill_robot_configs
from so101_nexus_maniskill.base_env import SO101NexusManiSkillBaseEnv

_DEFAULT_CONFIG = PickAndPlaceConfig()
PICK_AND_PLACE_CONFIGS: dict[str, dict] = build_maniskill_robot_configs(config=_DEFAULT_CONFIG)


@register_env("ManiSkillPickAndPlace-v1", max_episode_steps=_DEFAULT_CONFIG.max_episode_steps)
class PickAndPlaceEnv(SO101NexusManiSkillBaseEnv):
    """Pick-and-place environment with a visible coloured target disc on the ground."""

    config: PickAndPlaceConfig

    def __init__(
        self,
        *args,
        config: PickAndPlaceConfig = PickAndPlaceConfig(),
        robot_uids: str = "so100",
        num_envs: int = 1,
        reconfiguration_freq: int | None = None,
        **kwargs,
    ):
        self.cube_color_name = config.cube_color
        self.cube_color_rgba = CUBE_COLOR_MAP[config.cube_color]
        self.target_color_name = config.target_color
        self.target_color_rgba = TARGET_COLOR_MAP[config.target_color]
        self.cube_half_size = config.cube_half_size
        self.target_disc_radius = config.target_disc_radius
        self.task_description = (
            f"Pick up the small {config.cube_color} cube"
            f" and place it on the {config.target_color} circle"
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

        objs: list[Actor] = []
        for i in range(self.num_envs):
            cube = actors.build_cube(
                self.scene,
                half_size=self.cube_half_size,
                color=self.cube_color_rgba,
                name=f"cube-{i}",
                body_type="dynamic",
                scene_idxs=[i],
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
                material=sapien.render.RenderMaterial(base_color=self.target_color_rgba),
            )
            builder.initial_pose = sapien.Pose(p=[0, 0, 0], q=[0.7071068, 0.7071068, 0, 0])
            target = builder.build_kinematic(name=f"target_site-{i}")
            targets.append(target)
            self.remove_from_state_dict_registry(target)
        self.target_site = Actor.merge(targets, name="target_site")
        self.add_to_state_dict_registry(self.target_site)
        self._apply_robot_color_if_needed()

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict) -> None:
        with torch.device(self.device):
            b = len(env_idx)
            self._reset_robot(env_idx)

            cfg = self._robot_cfg
            spawn_cx, spawn_cy = cfg["cube_spawn_center"]
            spawn_hs = cfg["cube_spawn_half_size"]

            target_xyz = torch.zeros((b, 3), device=self.device)
            target_xyz[:, 0] = spawn_cx + (torch.rand(b, device=self.device) * 2 - 1) * spawn_hs
            target_xyz[:, 1] = spawn_cy + (torch.rand(b, device=self.device) * 2 - 1) * spawn_hs
            target_xyz[:, 2] = 0.001
            target_q = torch.tensor([[0.7071068, 0.7071068, 0.0, 0.0]], device=self.device).expand(
                b, -1
            )
            self.target_site.set_pose(Pose.create_from_pq(p=target_xyz, q=target_q))

            xyz = torch.zeros((b, 3), device=self.device)
            for _ in range(100):
                xyz[:, 0] = spawn_cx + (torch.rand(b, device=self.device) * 2 - 1) * spawn_hs
                xyz[:, 1] = spawn_cy + (torch.rand(b, device=self.device) * 2 - 1) * spawn_hs
                dists = torch.linalg.norm(xyz[:, :2] - target_xyz[:, :2], dim=1)
                if (dists >= self.config.min_cube_target_separation).all():
                    break
            xyz[:, 2] = self.cube_half_size
            qs = random_quaternions(b, lock_x=True, lock_y=True)
            self.obj.set_pose(Pose.create_from_pq(p=xyz, q=qs))
            self._store_initial_obj_z(env_idx, xyz[:, 2])

    def evaluate(self) -> dict[str, torch.Tensor]:
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
        obs = {
            "tcp_pose": self.agent.tcp_pose.raw_pose,
            "is_grasped": info["is_grasped"],
            "target_pos": self.target_site.pose.p,
        }
        if "state" in self.obs_mode:
            obs.update(
                {
                    "obj_pose": self.obj.pose.raw_pose,
                    "tcp_to_obj_pos": self.obj.pose.p - self.agent.tcp_pose.p,
                    "obj_to_target_pos": self.target_site.pose.p - self.obj.pose.p,
                }
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict) -> torch.Tensor:
        reach_progress = self._reach_progress(info["tcp_to_obj_dist"])
        is_grasped = info["is_grasped"]
        placement_progress = self._reach_progress(info["obj_to_target_dist"]) * is_grasped

        return self._assemble_normalized_reward(
            reach_progress=reach_progress,
            is_grasped=is_grasped,
            task_progress=placement_progress,
            is_complete=info["success"],
        )


def _register_robot_variant(
    *,
    class_name: str,
    env_id: str,
    robot_uid: str,
) -> type:
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("robot_uids", robot_uid)
        PickAndPlaceEnv.__init__(self, *args, **kwargs)

    cls = type(class_name, (PickAndPlaceEnv,), {"__init__": __init__})
    cls = register_env(env_id, max_episode_steps=_DEFAULT_CONFIG.max_episode_steps)(cls)
    globals()[class_name] = cls
    return cls


PickAndPlaceSO100Env = _register_robot_variant(
    class_name="PickAndPlaceSO100Env",
    env_id="ManiSkillPickAndPlaceSO100-v1",
    robot_uid="so100",
)
PickAndPlaceSO101Env = _register_robot_variant(
    class_name="PickAndPlaceSO101Env",
    env_id="ManiSkillPickAndPlaceSO101-v1",
    robot_uid="so101",
)
