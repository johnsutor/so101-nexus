from typing import Any, Literal, Union

import numpy as np
import sapien
import torch
from mani_skill.agents.robots.so100.so_100 import SO100
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils.randomization.pose import random_quaternions
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
from sapien.render import RenderBodyComponent
from transforms3d.euler import euler2quat

from so101_nexus_core.types import CUBE_COLOR_MAP, CubeColorName
from so101_nexus_maniskill.so101_agent import SO101

PICK_CUBE_CONFIGS: dict[str, dict] = {
    "so100": {
        "cube_half_size": 0.0125,
        "goal_thresh": 0.025,
        "cube_spawn_half_size": 0.05,
        "cube_spawn_center": (-0.46, 0),
        "max_goal_height": 0.08,
        "sensor_cam_eye_pos": [-0.27, 0, 0.4],
        "sensor_cam_target_pos": [-0.46, 0, 0.02],
        "human_cam_eye_pos": [0.0, 0.4, 0.4],
        "human_cam_target_pos": [-0.46, 0.0, 0.1],
        "wrist_camera_mount_link": "Fixed_Jaw",
    },
    "so101": {
        "cube_half_size": 0.0125,
        "goal_thresh": 0.025,
        "cube_spawn_half_size": 0.05,
        "cube_spawn_center": (-0.46, 0),
        "max_goal_height": 0.08,
        "sensor_cam_eye_pos": [-0.27, 0, 0.4],
        "sensor_cam_target_pos": [-0.46, 0, 0.02],
        "human_cam_eye_pos": [0.0, 0.4, 0.4],
        "human_cam_target_pos": [-0.46, 0.0, 0.1],
        "wrist_camera_mount_link": "gripper_link",
    },
}

CameraMode = Literal["fixed", "wrist", "both"]


@register_env("PickCubeGoal-v1", max_episode_steps=256)
class PickCubeEnv(BaseEnv):
    """Configurable pick-cube environment supporting SO100 and SO101 robots.

    The base class uses **goal-based** success: the cube must be placed within
    ``goal_thresh`` of a randomised goal site and the robot must be static.
    """

    SUPPORTED_ROBOTS = ["so100", "so101"]
    agent: Union[SO100, SO101]

    LIFT_THRESHOLD = 0.05

    def __init__(
        self,
        *args,
        robot_uids: str = "so100",
        cube_color: CubeColorName = "red",
        cube_half_size: float = 0.02,
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
        if not (0.01 <= cube_half_size <= 0.05):
            raise ValueError(f"cube_half_size must be in [0.01, 0.05], got {cube_half_size}")
        if robot_uids not in PICK_CUBE_CONFIGS:
            raise ValueError(
                f"robot_uids must be one of {list(PICK_CUBE_CONFIGS)}, got {robot_uids!r}"
            )

        self.cube_color_name = cube_color
        self.cube_color_rgba = CUBE_COLOR_MAP[cube_color]
        self.cube_half_size = cube_half_size
        self.robot_color = robot_color
        self.camera_mode: CameraMode = camera_mode
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.camera_width = camera_width
        self.camera_height = camera_height
        self._robot_cfg = PICK_CUBE_CONFIGS[robot_uids]

        self.task_description = f"pick up the small {cube_color} cube"

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
        """Return the default simulation configuration with increased contact buffers."""
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                max_rigid_contact_count=2**20, max_rigid_patch_count=2**19
            )
        )

    @property
    def _default_sensor_configs(self) -> list[CameraConfig]:
        """Return sensor camera configs based on the current ``camera_mode``.

        Returns a fixed overhead camera, a wrist-mounted camera, or both,
        depending on whether *camera_mode* is ``"fixed"``, ``"wrist"``, or
        ``"both"``.
        """
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
            roll_rad = np.radians(-180)
            pitch_rad = np.radians(np.random.uniform(-45, -30))
            yaw_rad = np.radians(-90)
            q = euler2quat(roll_rad, pitch_rad, yaw_rad, axes="sxyz")
            x = 0.0
            y = np.random.uniform(-0.06, -0.03)
            z = np.random.uniform(-0.06, -0.03)
            fov = np.random.uniform(np.pi / 3, np.pi / 2)
            configs.append(
                CameraConfig(
                    "wrist_camera",
                    sapien.Pose(p=[x, y, z], q=q),
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
        """Return a high-resolution camera config for human rendering."""
        cfg = self._robot_cfg
        pose = sapien_utils.look_at(cfg["human_cam_eye_pos"], cfg["human_cam_target_pos"])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict) -> None:
        """Load the robot agent at its fixed base position in the scene."""
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict) -> None:
        """Build the table scene, cube actors, goal site, and optional robot colouring."""
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

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

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict) -> None:
        """Randomise cube and goal poses for the environments in *env_idx*."""
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            cfg = self._robot_cfg
            spawn_cx, spawn_cy = cfg["cube_spawn_center"]
            spawn_hs = cfg["cube_spawn_half_size"]

            xyz = torch.zeros((b, 3))
            xyz[:, 0] = spawn_cx + (torch.rand(b) * 2 - 1) * spawn_hs
            xyz[:, 1] = spawn_cy + (torch.rand(b) * 2 - 1) * spawn_hs
            xyz[:, 2] = self.cube_half_size
            qs = random_quaternions(b, lock_x=True, lock_y=True)
            self.obj.set_pose(Pose.create_from_pq(p=xyz, q=qs))

            if self._initial_obj_z is None:
                self._initial_obj_z = torch.zeros(self.num_envs, device=self.device)
            self._initial_obj_z[env_idx] = xyz[:, 2]

            goal_xyz = torch.zeros((b, 3))
            goal_xyz[:, 0] = spawn_cx + (torch.rand(b) * 2 - 1) * spawn_hs
            goal_xyz[:, 1] = spawn_cy + (torch.rand(b) * 2 - 1) * spawn_hs
            goal_xyz[:, 2] = self.cube_half_size + torch.rand(b) * cfg["max_goal_height"]
            self.goal_site.set_pose(Pose.create_from_pq(p=goal_xyz))

    def evaluate(self) -> dict[str, torch.Tensor]:
        """Compute per-environment success metrics.

        Returns
        -------
        dict
            A dict with keys ``obj_to_goal_dist``, ``is_obj_placed``,
            ``is_grasped``, ``is_robot_static``, ``lift_height``, and
            ``success``, each a tensor of shape ``(num_envs,)``.
        """
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
        """Return task-specific observation extras.

        Always includes TCP pose, grasp flag, and goal position.  When
        ``obs_mode`` contains ``"state"``, object pose and relative positions
        are also included.
        """
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
        """Compute a shaped dense reward for the goal-based task.

        The reward is composed of: reaching (1), grasping (+1), placement (+1
        when grasped), static bonus (+1 when placed), and a success bonus of
        5.0 when the episode is solved.

        Returns
        -------
        torch.Tensor
            Reward tensor of shape ``(num_envs,)``.
        """
        tcp_to_obj_dist = torch.linalg.norm(self.obj.pose.p - self.agent.tcp_pose.p, axis=1)
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)

        reward = reaching_reward

        is_grasped = info["is_grasped"]
        reward = reward + is_grasped

        obj_to_goal_dist = info["obj_to_goal_dist"]
        placement_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
        reward = reward + placement_reward * is_grasped

        is_robot_static = info["is_robot_static"]
        reward = reward + is_robot_static * info["is_obj_placed"]

        reward[info["success"]] = 5.0

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ) -> torch.Tensor:
        """Return the dense reward normalised to ``[0, 1]``."""
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5.0


PickCubeGoalEnv = PickCubeEnv


@register_env("PickCubeLift-v1", max_episode_steps=256)
class PickCubeLiftEnv(PickCubeEnv):
    """Pick-cube with lift-based success: cube must be lifted above threshold while grasped."""

    def evaluate(self) -> dict[str, torch.Tensor]:
        """Compute per-environment success metrics for the lift task.

        Success requires the cube to be lifted above ``LIFT_THRESHOLD`` metres
        while actively grasped.

        Returns
        -------
        dict
            A dict with keys ``obj_to_goal_dist``, ``is_obj_placed``,
            ``is_grasped``, ``is_robot_static``, ``lift_height``, and
            ``success``, each a tensor of shape ``(num_envs,)``.
        """
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
        """Compute a shaped dense reward for the lift-based task.

        The reward is composed of: reaching (1), grasping (+1), lifting (+1
        when grasped), and a success bonus of 6.0 when the episode is solved.

        Returns
        -------
        torch.Tensor
            Reward tensor of shape ``(num_envs,)``.
        """
        tcp_to_obj_dist = torch.linalg.norm(self.obj.pose.p - self.agent.tcp_pose.p, axis=1)
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)

        reward = reaching_reward

        is_grasped = info["is_grasped"]
        reward = reward + is_grasped

        lift_height = info["lift_height"].clamp(min=0.0)
        lift_reward = torch.tanh(5 * lift_height)
        reward = reward + lift_reward * is_grasped

        reward[info["success"]] = 6.0

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ) -> torch.Tensor:
        """Return the dense reward normalised to ``[0, 1]``."""
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 6.0


@register_env("PickCubeGoalSO100-v1", max_episode_steps=256)
class PickCubeGoalSO100Env(PickCubeEnv):
    """Goal-based pick-cube pre-configured for SO100."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("robot_uids", "so100")
        super().__init__(*args, **kwargs)


@register_env("PickCubeGoalSO101-v1", max_episode_steps=256)
class PickCubeGoalSO101Env(PickCubeEnv):
    """Goal-based pick-cube pre-configured for SO101."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("robot_uids", "so101")
        super().__init__(*args, **kwargs)


@register_env("PickCubeLiftSO100-v1", max_episode_steps=256)
class PickCubeLiftSO100Env(PickCubeLiftEnv):
    """Lift-based pick-cube pre-configured for SO100."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("robot_uids", "so100")
        super().__init__(*args, **kwargs)


@register_env("PickCubeLiftSO101-v1", max_episode_steps=256)
class PickCubeLiftSO101Env(PickCubeLiftEnv):
    """Lift-based pick-cube pre-configured for SO101."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("robot_uids", "so101")
        super().__init__(*args, **kwargs)
