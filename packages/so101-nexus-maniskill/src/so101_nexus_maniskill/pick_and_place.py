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
    CUBE_COLOR_MAP,
    DEFAULT_CUBE_HALF_SIZE,
    DEFAULT_CUBE_SPAWN_HALF_SIZE,
    DEFAULT_GOAL_THRESH,
    DEFAULT_GROUND_COLOR,
    DEFAULT_MIN_CUBE_TARGET_SEPARATION,
    DEFAULT_TARGET_DISC_RADIUS,
    REWARD_WEIGHT_COMPLETION_BONUS,
    REWARD_WEIGHT_GRASPING,
    REWARD_WEIGHT_REACHING,
    REWARD_WEIGHT_TASK_OBJECTIVE,
    TARGET_COLOR_MAP,
    CubeColorName,
    TargetColorName,
)
from so101_nexus_maniskill.so101_agent import SO101

PICK_AND_PLACE_CONFIGS: dict[str, dict] = {
    "so100": {
        "cube_half_size": DEFAULT_CUBE_HALF_SIZE,
        "goal_thresh": DEFAULT_GOAL_THRESH,
        "cube_spawn_half_size": DEFAULT_CUBE_SPAWN_HALF_SIZE,
        "cube_spawn_center": (0.15, 0),
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
        "cube_half_size": DEFAULT_CUBE_HALF_SIZE,
        "goal_thresh": DEFAULT_GOAL_THRESH,
        "cube_spawn_half_size": DEFAULT_CUBE_SPAWN_HALF_SIZE,
        "cube_spawn_center": (0.15, 0),
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


@register_env("ManiSkillPickAndPlace-v1", max_episode_steps=256)
class PickAndPlaceEnv(BaseEnv):
    """Pick-and-place environment with a visible coloured target disc on the ground.

    The agent must pick up a coloured cube and place it on a differently-coloured
    disc. Success requires XY proximity to the target and the cube being near
    ground level while the robot is static.
    """

    SUPPORTED_ROBOTS = ["so100", "so101"]
    agent: Union[SO100, SO101]

    def __init__(
        self,
        *args,
        robot_uids: str = "so100",
        cube_color: CubeColorName = "red",
        target_color: TargetColorName = "blue",
        cube_half_size: float = DEFAULT_CUBE_HALF_SIZE,
        robot_color: tuple[float, float, float, float] | None = None,
        camera_mode: CameraMode = "fixed",
        robot_init_qpos_noise: float = 0.02,
        num_envs: int = 1,
        reconfiguration_freq: int | None = None,
        camera_width: int = 224,
        camera_height: int = 224,
        **kwargs,
    ):
        if cube_color not in CUBE_COLOR_MAP:
            raise ValueError(
                f"cube_color must be one of {list(CUBE_COLOR_MAP)}, got {cube_color!r}"
            )
        if target_color not in TARGET_COLOR_MAP:
            raise ValueError(
                f"target_color must be one of {list(TARGET_COLOR_MAP)}, got {target_color!r}"
            )
        if cube_color == target_color:
            raise ValueError(f"cube_color and target_color must differ, both are {cube_color!r}")
        if not (0.01 <= cube_half_size <= 0.05):
            raise ValueError(f"cube_half_size must be in [0.01, 0.05], got {cube_half_size}")
        if robot_uids not in PICK_AND_PLACE_CONFIGS:
            raise ValueError(
                f"robot_uids must be one of {list(PICK_AND_PLACE_CONFIGS)}, got {robot_uids!r}"
            )

        self.cube_color_name = cube_color
        self.cube_color_rgba = CUBE_COLOR_MAP[cube_color]
        self.target_color_name = target_color
        self.target_color_rgba = TARGET_COLOR_MAP[target_color]
        self.cube_half_size = cube_half_size
        self.target_disc_radius = DEFAULT_TARGET_DISC_RADIUS
        self.robot_color = robot_color
        self.camera_mode: CameraMode = camera_mode
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.camera_width = camera_width
        self.camera_height = camera_height
        self._robot_cfg = PICK_AND_PLACE_CONFIGS[robot_uids]

        self.task_description = (
            f"Pick up the small {cube_color} cube and place it on the {target_color} circle"
        )

        if reconfiguration_freq is None:
            reconfiguration_freq = 0

        self._initial_obj_z: torch.Tensor | None = None

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
            cube = actors.build_cube(
                self.scene,
                half_size=self.cube_half_size,
                color=self.cube_color_rgba,
                name=f"cube-{i}",
                body_type="dynamic",
                scene_idxs=[i],
                initial_pose=sapien.Pose(p=[0, 0, 0]),
            )
            self._objs.append(cube)
            self.remove_from_state_dict_registry(cube)
        self.obj = Actor.merge(self._objs, name="cube")
        self.add_to_state_dict_registry(self.obj)

        target_rgba = self.target_color_rgba
        target_builders: list[Actor] = []
        for i in range(self.num_envs):
            builder = self.scene.create_actor_builder()
            builder.add_cylinder_visual(
                radius=self.target_disc_radius,
                pose=sapien.Pose(p=[0, 0, 0.001], q=[0, 0.7071068, 0.7071068, 0 ]),
                half_length=0.001,
                material=sapien.render.RenderMaterial(
                    base_color=target_rgba,
                ),
            )
            builder.initial_pose = sapien.Pose(p=[0, 0, 0], q=[0.7071068, 0.7071068, 0, 0])
            target = builder.build_kinematic(name=f"target_site-{i}")
            target_builders.append(target)
            self.remove_from_state_dict_registry(target)
        self.target_site = Actor.merge(target_builders, name="target_site")
        self.add_to_state_dict_registry(self.target_site)

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

            target_xyz = torch.zeros((b, 3))
            target_xyz[:, 0] = spawn_cx + (torch.rand(b) * 2 - 1) * spawn_hs
            target_xyz[:, 1] = spawn_cy + (torch.rand(b) * 2 - 1) * spawn_hs
            target_xyz[:, 2] = 0.001
            target_q = torch.tensor([[0.7071068, 0.7071068, 0, 0]], dtype=torch.float32).expand(
                b, -1
            )
            self.target_site.set_pose(Pose.create_from_pq(p=target_xyz, q=target_q))

            xyz = torch.zeros((b, 3))
            for attempt in range(100):
                xyz[:, 0] = spawn_cx + (torch.rand(b) * 2 - 1) * spawn_hs
                xyz[:, 1] = spawn_cy + (torch.rand(b) * 2 - 1) * spawn_hs
                dists = torch.linalg.norm(xyz[:, :2] - target_xyz[:, :2], dim=1)
                if (dists >= DEFAULT_MIN_CUBE_TARGET_SEPARATION).all():
                    break
            xyz[:, 2] = self.cube_half_size
            qs = random_quaternions(b, lock_x=True, lock_y=True)
            self.obj.set_pose(Pose.create_from_pq(p=xyz, q=qs))

            if self._initial_obj_z is None:
                self._initial_obj_z = torch.zeros(self.num_envs, device=self.device)
            self._initial_obj_z[env_idx] = xyz[:, 2]

    def evaluate(self) -> dict[str, torch.Tensor]:
        """Evaluate the current state of the environment.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary containing evaluation metrics including distances,
            grasp state, and success.
        """
        obj_to_target_xy = self.obj.pose.p[:, :2] - self.target_site.pose.p[:, :2]
        obj_to_target_dist = torch.linalg.norm(obj_to_target_xy, axis=1)
        cube_near_ground = self.obj.pose.p[:, 2] < (self.cube_half_size + 0.01)
        is_obj_placed = (obj_to_target_dist <= self._robot_cfg["goal_thresh"]) & cube_near_ground
        is_grasped = self.agent.is_grasping(self.obj)
        is_robot_static = self.agent.is_static()

        obj_z = self.obj.pose.p[:, 2]
        lift_height = obj_z - self._initial_obj_z

        success = is_obj_placed & is_robot_static

        return dict(
            obj_to_target_dist=obj_to_target_dist,
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
            target_pos=self.target_site.pose.p,
        )
        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.obj.pose.raw_pose,
                tcp_to_obj_pos=self.obj.pose.p - self.agent.tcp_pose.p,
                obj_to_target_pos=self.target_site.pose.p - self.obj.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict) -> torch.Tensor:
        """Compute dense reward for the pick-and-place task.

        Parameters
        ----------
        obs : Any
            Current observation.
        action : torch.Tensor
            Action taken.
        info : dict
            Info dict from evaluate().

        Returns
        -------
        torch.Tensor
            Reward tensor in [0, 1].
        """
        tcp_to_obj_dist = torch.linalg.norm(self.obj.pose.p - self.agent.tcp_pose.p, axis=1)
        reach_progress = 1 - torch.tanh(5 * tcp_to_obj_dist)

        is_grasped = info["is_grasped"]

        obj_to_target_dist = info["obj_to_target_dist"]
        placement_progress = (1 - torch.tanh(5 * obj_to_target_dist)) * is_grasped

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


@register_env("ManiSkillPickAndPlaceSO100-v1", max_episode_steps=256)
class PickAndPlaceSO100Env(PickAndPlaceEnv):
    """Pick-and-place pre-configured for SO100."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("robot_uids", "so100")
        super().__init__(*args, **kwargs)


@register_env("ManiSkillPickAndPlaceSO101-v1", max_episode_steps=256)
class PickAndPlaceSO101Env(PickAndPlaceEnv):
    """Pick-and-place pre-configured for SO101."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("robot_uids", "so101")
        super().__init__(*args, **kwargs)
