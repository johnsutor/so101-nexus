from typing import Any

import sapien
import torch
from mani_skill.envs.utils.randomization.pose import random_quaternions
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose

from so101_nexus_core.robot_presets import build_maniskill_robot_configs
from so101_nexus_core.types import (
    DEFAULT_CAMERA_HEIGHT,
    DEFAULT_CAMERA_WIDTH,
    DEFAULT_LIFT_THRESHOLD,
    DEFAULT_MAX_EPISODE_STEPS,
    YCB_ENV_NAME_MAP,
    YCB_OBJECTS,
    YcbModelId,
)
from so101_nexus_core.ycb_geometry import get_maniskill_ycb_spawn_z
from so101_nexus_maniskill.base_env import CameraMode, SO101NexusManiSkillBaseEnv

PICK_YCB_CONFIGS: dict[str, dict] = build_maniskill_robot_configs(
    include_cube_half_size=False,
    include_max_goal_height=True,
)


@register_env("ManiSkillPickYCBGoal-v1", max_episode_steps=DEFAULT_MAX_EPISODE_STEPS)
class PickYCBEnv(SO101NexusManiSkillBaseEnv):
    """Configurable pick-YCB environment supporting SO100 and SO101 robots."""

    LIFT_THRESHOLD = DEFAULT_LIFT_THRESHOLD

    def __init__(
        self,
        *args,
        robot_uids: str = "so100",
        model_id: YcbModelId = "058_golf_ball",
        robot_color: tuple[float, float, float, float] | None = None,
        camera_mode: CameraMode = "fixed",
        robot_init_qpos_noise: float = 0.02,
        num_envs: int = 1,
        reconfiguration_freq: int | None = None,
        camera_width: int = DEFAULT_CAMERA_WIDTH,
        camera_height: int = DEFAULT_CAMERA_HEIGHT,
        **kwargs,
    ):
        if model_id not in YCB_OBJECTS:
            raise ValueError(f"model_id must be one of {list(YCB_OBJECTS)}, got {model_id!r}")

        self.model_id = model_id
        self._obj_spawn_z: float | None = None
        self.task_description = f"Pick up the {YCB_OBJECTS[model_id]}"

        self._setup_base(
            robot_uids=robot_uids,
            robot_cfgs=PICK_YCB_CONFIGS,
            robot_color=robot_color,
            camera_mode=camera_mode,
            robot_init_qpos_noise=robot_init_qpos_noise,
            camera_width=camera_width,
            camera_height=camera_height,
        )

        if reconfiguration_freq is None:
            reconfiguration_freq = 1 if camera_mode in ("wrist", "both") else 0

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

    def _load_scene(self, options: dict) -> None:
        self._build_ground()

        objs: list[Actor] = []
        for i in range(self.num_envs):
            builder = actors.get_actor_builder(self.scene, id=f"ycb:{self.model_id}")
            builder.initial_pose = sapien.Pose(p=[0, 0, 0])
            builder.set_scene_idxs([i])
            obj = builder.build(name=f"ycb_obj-{i}")
            objs.append(obj)
            self.remove_from_state_dict_registry(obj)
        self.obj = Actor.merge(objs, name="ycb_obj")
        self.add_to_state_dict_registry(self.obj)

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

            xyz = torch.zeros((b, 3), device=self.device)
            xyz[:, 0] = spawn_cx + (torch.rand(b, device=self.device) * 2 - 1) * spawn_hs
            xyz[:, 1] = spawn_cy + (torch.rand(b, device=self.device) * 2 - 1) * spawn_hs
            xyz[:, 2] = self._obj_spawn_z
            qs = random_quaternions(b, lock_x=True, lock_y=True)
            self.obj.set_pose(Pose.create_from_pq(p=xyz, q=qs))
            self._store_initial_obj_z(env_idx, xyz[:, 2])

            goal_xyz = torch.zeros((b, 3), device=self.device)
            goal_xyz[:, 0] = spawn_cx + (torch.rand(b, device=self.device) * 2 - 1) * spawn_hs
            goal_xyz[:, 1] = spawn_cy + (torch.rand(b, device=self.device) * 2 - 1) * spawn_hs
            goal_xyz[:, 2] = (
                self._obj_spawn_z + torch.rand(b, device=self.device) * cfg["max_goal_height"]
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


PickYCBGoalEnv = PickYCBEnv


@register_env("ManiSkillPickYCBLift-v1", max_episode_steps=DEFAULT_MAX_EPISODE_STEPS)
class PickYCBLiftEnv(PickYCBEnv):
    """Pick-YCB variant where success is lift height threshold while grasped."""

    def evaluate(self) -> dict[str, torch.Tensor]:
        info = super().evaluate()
        info["success"] = (info["lift_height"] > self.LIFT_THRESHOLD) & info["is_grasped"]
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


for _model_id, _env_name in YCB_ENV_NAME_MAP.items():
    for _task, _base_cls in [("Goal", PickYCBEnv), ("Lift", PickYCBLiftEnv)]:
        for _robot in ["SO100", "SO101"]:
            _env_id = f"ManiSkillPick{_env_name}{_task}{_robot}-v1"
            _robot_uid = _robot.lower()

            def _make_init(_mid=_model_id, _ruid=_robot_uid, _base=_base_cls):
                def __init__(self, *args, **kwargs):
                    kwargs.setdefault("robot_uids", _ruid)
                    kwargs.setdefault("model_id", _mid)
                    _base.__init__(self, *args, **kwargs)

                return __init__

            _cls = type(
                f"Pick{_env_name}{_task}{_robot}Env",
                (_base_cls,),
                {"__init__": _make_init()},
            )
            _cls = register_env(_env_id, max_episode_steps=DEFAULT_MAX_EPISODE_STEPS)(_cls)
            globals()[f"Pick{_env_name}{_task}{_robot}Env"] = _cls
