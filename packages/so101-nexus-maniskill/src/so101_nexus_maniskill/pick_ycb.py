from typing import Any, Literal, Union

import numpy as np
import sapien
import sapien.render
import torch
from mani_skill.agents.robots.so100.so_100 import SO100
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils.randomization.pose import random_quaternions
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
from sapien.render import RenderBodyComponent
from transforms3d.euler import euler2quat

from so101_nexus_core.types import (
    DEFAULT_CUBE_SPAWN_HALF_SIZE,
    DEFAULT_GOAL_THRESH,
    DEFAULT_GROUND_COLOR,
    DEFAULT_LIFT_THRESHOLD,
    DEFAULT_MAX_GOAL_HEIGHT,
    REWARD_WEIGHT_COMPLETION_BONUS,
    REWARD_WEIGHT_GRASPING,
    REWARD_WEIGHT_REACHING,
    REWARD_WEIGHT_TASK_OBJECTIVE,
    YCB_OBJECTS,
)
from so101_nexus_maniskill.so101_agent import SO101

PICK_YCB_CONFIGS: dict[str, dict] = {
    "so100": {
        "goal_thresh": DEFAULT_GOAL_THRESH,
        "cube_spawn_half_size": DEFAULT_CUBE_SPAWN_HALF_SIZE,
        "cube_spawn_center": (0.15, 0),
        "max_goal_height": DEFAULT_MAX_GOAL_HEIGHT,
        "base_quat": euler2quat(0, 0, np.pi / 2),
        "sensor_cam_eye_pos": [0.0, 0.3, 0.3],
        "sensor_cam_target_pos": [0.15, 0, 0.02],
        "human_cam_eye_pos": [0.0, 0.4, 0.4],
        "human_cam_target_pos": [0.15, 0.0, 0.05],
        "wrist_camera_mount_link": "Fixed_Jaw",
        "wrist_cam_pos_center": [0.0, -0.045, -0.045],
        "wrist_cam_pos_noise": [0.0, 0.015, 0.015],
        "wrist_cam_euler_center": [-np.pi, np.radians(-37.5), np.radians(-90)],
        "wrist_cam_euler_noise": [0.0, np.radians(7.5), 0.0],
        "wrist_cam_fov_range": [np.pi / 3, np.pi / 2],
    },
    "so101": {
        "goal_thresh": DEFAULT_GOAL_THRESH,
        "cube_spawn_half_size": DEFAULT_CUBE_SPAWN_HALF_SIZE,
        "cube_spawn_center": (0.15, 0),
        "max_goal_height": DEFAULT_MAX_GOAL_HEIGHT,
        "base_quat": euler2quat(0, 0, 0),
        "sensor_cam_eye_pos": [0.0, 0.3, 0.3],
        "sensor_cam_target_pos": [0.15, 0, 0.02],
        "human_cam_eye_pos": [0.0, 0.4, 0.4],
        "human_cam_target_pos": [0.15, 0.0, 0.05],
        "wrist_camera_mount_link": "gripper_link",
        "wrist_cam_pos_center": [0.0, 0.04, -0.04],
        "wrist_cam_pos_noise": [0.005, 0.01, 0.01],
        "wrist_cam_euler_center": [-np.pi, np.radians(37.5), np.radians(-90)],
        "wrist_cam_euler_noise": [0.0, 0.2, 0.0],
        "wrist_cam_fov_range": [np.pi / 3, np.pi / 2],
    },
}

CameraMode = Literal["fixed", "wrist", "both"]


@register_env("ManiSkillPickYCBGoal-v1", max_episode_steps=256)
class PickYCBEnv(BaseEnv):
    """Configurable pick-YCB environment supporting SO100 and SO101 robots.

    Goal-based success: the object must be placed within ``goal_thresh``
    of a randomised goal site and the robot must be static.
    """

    SUPPORTED_ROBOTS = ["so100", "so101"]
    agent: Union[SO100, SO101]

    LIFT_THRESHOLD = DEFAULT_LIFT_THRESHOLD

    def __init__(
        self,
        *args,
        robot_uids: str = "so100",
        model_id: str = "058_golf_ball",
        robot_color: tuple[float, float, float, float] | None = None,
        camera_mode: CameraMode = "fixed",
        robot_init_qpos_noise: float = 0.02,
        num_envs: int = 1,
        reconfiguration_freq: int | None = None,
        camera_width: int = 224,
        camera_height: int = 224,
        **kwargs,
    ):
        if model_id not in YCB_OBJECTS:
            raise ValueError(f"model_id must be one of {list(YCB_OBJECTS)}, got {model_id!r}")
        if robot_uids not in PICK_YCB_CONFIGS:
            raise ValueError(
                f"robot_uids must be one of {list(PICK_YCB_CONFIGS)}, got {robot_uids!r}"
            )

        self.model_id = model_id
        self.robot_color = robot_color
        self.camera_mode: CameraMode = camera_mode
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.camera_width = camera_width
        self.camera_height = camera_height
        self._robot_cfg = PICK_YCB_CONFIGS[robot_uids]

        self.task_description = f"pick up the {YCB_OBJECTS[model_id]}"

        if reconfiguration_freq is None:
            reconfiguration_freq = 0

        self._initial_obj_z: torch.Tensor | None = None
        self._obj_spawn_z: float | None = None

        super().__init__(
            *args,
            robot_uids=robot_uids,
            reconfiguration_freq=reconfiguration_freq,
            num_envs=num_envs,
            **kwargs,
        )

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

        if self.camera_mode in ("fixed", "both"):
            pose = sapien_utils.look_at(cfg["sensor_cam_eye_pos"], cfg["sensor_cam_target_pos"])
            configs.append(
                CameraConfig(
                    "base_camera",
                    pose,
                    self.camera_width,
                    self.camera_height,
                    np.pi / 3,
                    0.01,
                    100,
                )
            )

        if self.camera_mode in ("wrist", "both"):
            mount_link = self.agent.robot.links_map[cfg["wrist_camera_mount_link"]]
            pos_c = cfg["wrist_cam_pos_center"]
            pos_n = cfg["wrist_cam_pos_noise"]
            eul_c = cfg["wrist_cam_euler_center"]
            eul_n = cfg["wrist_cam_euler_noise"]
            fov_lo, fov_hi = cfg["wrist_cam_fov_range"]

            p = [c + np.random.uniform(-n, n) for c, n in zip(pos_c, pos_n)]
            e = [c + np.random.uniform(-n, n) for c, n in zip(eul_c, eul_n)]
            q = euler2quat(*e, axes="sxyz")
            fov = np.random.uniform(fov_lo, fov_hi)

            configs.append(
                CameraConfig(
                    "wrist_camera",
                    sapien.Pose(p=p, q=q),
                    self.camera_width,
                    self.camera_height,
                    fov,
                    0.01,
                    100,
                    mount=mount_link,
                )
            )

        return configs

    @property
    def _default_human_render_camera_configs(self) -> CameraConfig:
        cfg = self._robot_cfg
        pose = sapien_utils.look_at(cfg["human_cam_eye_pos"], cfg["human_cam_target_pos"])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict) -> None:
        super()._load_agent(options, sapien.Pose(p=[0, 0, 0], q=self._robot_cfg["base_quat"]))

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

    def _load_scene(self, options: dict) -> None:
        ground_builder = self.scene.create_actor_builder()
        ground_builder.add_plane_collision(
            sapien.Pose(p=[0, 0, 0], q=[0.7071068, 0, -0.7071068, 0])
        )
        ground_builder.initial_pose = sapien.Pose(p=[0, 0, 0])
        if self.scene.parallel_in_single_scene:
            ground_builder.set_scene_idxs([0])
        ground = ground_builder.build_static(name="ground")

        if self.scene.can_render():
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
            mat.base_color = DEFAULT_GROUND_COLOR
            shape = sapien.render.RenderShapeTriangleMesh(
                vertices=verts, triangles=tris, normals=normals, uvs=uvs, material=mat
            )
            for obj in ground._objs:
                comp = sapien.render.RenderBodyComponent()
                comp.attach(shape)
                obj.add_component(comp)

        self._objs: list[Actor] = []
        for i in range(self.num_envs):
            builder = actors.get_actor_builder(self.scene, id=f"ycb:{self.model_id}")
            builder.initial_pose = sapien.Pose(p=[0, 0, 0])
            builder.set_scene_idxs([i])
            obj = builder.build(name=f"ycb_obj-{i}")
            self._objs.append(obj)
            self.remove_from_state_dict_registry(obj)
        self.obj = Actor.merge(self._objs, name="ycb_obj")
        self.add_to_state_dict_registry(self.obj)

        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self._robot_cfg["goal_thresh"],
            color=[0, 1, 0, 0.5],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(p=[0, 0, 0]),
        )
        self._hidden_objects.append(self.goal_site)

        if self.robot_color is not None:
            color = list(self.robot_color)
            for link in self.agent.robot.links:
                for obj in link._objs:
                    render_body: RenderBodyComponent = obj.entity.find_component_by_type(
                        RenderBodyComponent
                    )
                    if render_body is not None:
                        for render_shape in render_body.render_shapes:
                            for part in render_shape.parts:
                                part.material.set_base_color(color)

    def _after_reconfigure(self, options: dict) -> None:
        # Compute initial Z from collision mesh AABB bounds
        import json
        from pathlib import Path

        info_path = (
            Path.home() / ".maniskill" / "data" / "assets" / "mani_skill2_ycb" / "info_pick_v0.json"
        )
        with open(info_path) as f:
            model_db = json.load(f)
        metadata = model_db[self.model_id]
        scales = metadata.get("scales", [1.0])
        scale = scales[0]
        bbox_min_z = metadata["bbox"]["min"][2] * scale
        # Spawn Z places the bottom of the bounding box on the ground
        # with a small margin to avoid interpenetration.
        self._obj_spawn_z = -bbox_min_z + 0.002

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict) -> None:
        with torch.device(self.device):
            b = len(env_idx)

            qpos = (
                torch.tensor(self.agent.keyframes["rest"].qpos, dtype=torch.float32)
                .unsqueeze(0)
                .expand(b, -1)
                .clone()
            )
            noise = (torch.rand_like(qpos) * 2 - 1) * self.robot_init_qpos_noise
            self.agent.reset(qpos + noise)
            self.agent.robot.set_pose(sapien.Pose(p=[0, 0, 0], q=self._robot_cfg["base_quat"]))

            cfg = self._robot_cfg
            spawn_cx, spawn_cy = cfg["cube_spawn_center"]
            spawn_hs = cfg["cube_spawn_half_size"]

            xyz = torch.zeros((b, 3))
            xyz[:, 0] = spawn_cx + (torch.rand(b) * 2 - 1) * spawn_hs
            xyz[:, 1] = spawn_cy + (torch.rand(b) * 2 - 1) * spawn_hs
            xyz[:, 2] = self._obj_spawn_z
            qs = random_quaternions(b, lock_x=True, lock_y=True)
            self.obj.set_pose(Pose.create_from_pq(p=xyz, q=qs))

            if self._initial_obj_z is None:
                self._initial_obj_z = torch.zeros(self.num_envs, device=self.device)
            self._initial_obj_z[env_idx] = xyz[:, 2]

            goal_xyz = torch.zeros((b, 3))
            goal_xyz[:, 0] = spawn_cx + (torch.rand(b) * 2 - 1) * spawn_hs
            goal_xyz[:, 1] = spawn_cy + (torch.rand(b) * 2 - 1) * spawn_hs
            goal_xyz[:, 2] = self._obj_spawn_z + torch.rand(b) * cfg["max_goal_height"]
            self.goal_site.set_pose(Pose.create_from_pq(p=goal_xyz))

    def evaluate(self) -> dict[str, torch.Tensor]:
        obj_to_goal_dist = torch.linalg.norm(self.obj.pose.p - self.goal_site.pose.p, axis=1)
        is_obj_placed = obj_to_goal_dist <= self._robot_cfg["goal_thresh"]
        is_grasped = self.agent.is_grasping(self.obj)
        is_robot_static = self.agent.is_static()

        obj_z = self.obj.pose.p[:, 2]
        lift_height = obj_z - self._initial_obj_z

        success = is_obj_placed & is_robot_static

        return dict(
            obj_to_goal_dist=obj_to_goal_dist,
            is_obj_placed=is_obj_placed,
            is_grasped=is_grasped,
            is_robot_static=is_robot_static,
            lift_height=lift_height,
            success=success,
        )

    def _get_obs_extra(self, info: dict) -> dict[str, torch.Tensor]:
        obs = dict(
            tcp_pose=self.agent.tcp_pose.raw_pose,
            is_grasped=info["is_grasped"],
            goal_pos=self.goal_site.pose.p,
        )
        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.obj.pose.raw_pose,
                tcp_to_obj_pos=self.obj.pose.p - self.agent.tcp_pose.p,
                obj_to_goal_pos=self.goal_site.pose.p - self.obj.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict) -> torch.Tensor:
        tcp_to_obj_dist = torch.linalg.norm(self.obj.pose.p - self.agent.tcp_pose.p, axis=1)
        reach_progress = 1 - torch.tanh(5 * tcp_to_obj_dist)

        is_grasped = info["is_grasped"]

        obj_to_goal_dist = info["obj_to_goal_dist"]
        placement_progress = (1 - torch.tanh(5 * obj_to_goal_dist)) * is_grasped

        is_complete = info["success"] & info["is_robot_static"]

        reward = (
            REWARD_WEIGHT_REACHING * reach_progress
            + REWARD_WEIGHT_GRASPING * is_grasped
            + REWARD_WEIGHT_TASK_OBJECTIVE * placement_progress
            + REWARD_WEIGHT_COMPLETION_BONUS * is_complete
        )

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ) -> torch.Tensor:
        return self.compute_dense_reward(obs=obs, action=action, info=info)


PickYCBGoalEnv = PickYCBEnv


@register_env("ManiSkillPickYCBLift-v1", max_episode_steps=256)
class PickYCBLiftEnv(PickYCBEnv):
    """Pick-YCB with lift-based success: object must be lifted above threshold while grasped."""

    def evaluate(self) -> dict[str, torch.Tensor]:
        is_grasped = self.agent.is_grasping(self.obj)
        is_robot_static = self.agent.is_static()

        obj_z = self.obj.pose.p[:, 2]
        lift_height = obj_z - self._initial_obj_z

        obj_to_goal_dist = torch.linalg.norm(self.obj.pose.p - self.goal_site.pose.p, axis=1)
        is_obj_placed = obj_to_goal_dist <= self._robot_cfg["goal_thresh"]

        success = (lift_height > self.LIFT_THRESHOLD) & is_grasped

        return dict(
            obj_to_goal_dist=obj_to_goal_dist,
            is_obj_placed=is_obj_placed,
            is_grasped=is_grasped,
            is_robot_static=is_robot_static,
            lift_height=lift_height,
            success=success,
        )

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict) -> torch.Tensor:
        tcp_to_obj_dist = torch.linalg.norm(self.obj.pose.p - self.agent.tcp_pose.p, axis=1)
        reach_progress = 1 - torch.tanh(5 * tcp_to_obj_dist)

        is_grasped = info["is_grasped"]

        lift_height = info["lift_height"].clamp(min=0.0)
        lift_progress = torch.tanh(5 * lift_height) * is_grasped

        is_complete = info["success"]

        reward = (
            REWARD_WEIGHT_REACHING * reach_progress
            + REWARD_WEIGHT_GRASPING * is_grasped
            + REWARD_WEIGHT_TASK_OBJECTIVE * lift_progress
            + REWARD_WEIGHT_COMPLETION_BONUS * is_complete
        )

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ) -> torch.Tensor:
        return self.compute_dense_reward(obs=obs, action=action, info=info)


# --- Factory-generated per-object, per-robot subclasses ---

_YCB_ENV_NAME_MAP = {
    "009_gelatin_box": "GelatinBox",
    "011_banana": "Banana",
    "030_fork": "Fork",
    "031_spoon": "Spoon",
    "032_knife": "Knife",
    "033_spatula": "Spatula",
    "037_scissors": "Scissors",
    "040_large_marker": "LargeMarker",
    "043_phillips_screwdriver": "PhillipsScrewdriver",
    "058_golf_ball": "GolfBall",
}

# Generate subclasses for each (object, task, robot) combination
for _model_id, _env_name in _YCB_ENV_NAME_MAP.items():
    for _task, _base_cls in [("Goal", PickYCBEnv), ("Lift", PickYCBLiftEnv)]:
        for _robot in ["SO100", "SO101"]:
            _env_id = f"ManiSkillPick{_env_name}{_task}{_robot}-v1"
            _robot_uid = _robot.lower()

            # Create a closure to capture loop variables
            def _make_init(_mid=_model_id, _ruid=_robot_uid):
                def __init__(self, *args, **kwargs):
                    kwargs.setdefault("robot_uids", _ruid)
                    kwargs.setdefault("model_id", _mid)
                    super(type(self), self).__init__(*args, **kwargs)

                return __init__

            _cls = type(
                f"Pick{_env_name}{_task}{_robot}Env",
                (_base_cls,),
                {"__init__": _make_init()},
            )
            _cls = register_env(_env_id, max_episode_steps=256)(_cls)

            # Add to module globals for importability
            globals()[f"Pick{_env_name}{_task}{_robot}Env"] = _cls
