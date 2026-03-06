from __future__ import annotations

from typing import Literal, Union

import numpy as np
import sapien
import sapien.render
import torch
from mani_skill.agents.robots.so100.so_100 import SO100
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
from sapien.render import RenderBodyComponent
from transforms3d.euler import euler2quat

from so101_nexus_core.config import EnvironmentConfig
from so101_nexus_maniskill.so101_agent import SO101

CameraMode = Literal["fixed", "wrist", "both"]


class SO101NexusManiSkillBaseEnv(BaseEnv):
    """Shared ManiSkill base class for SO101-Nexus tasks."""

    SUPPORTED_ROBOTS = ["so100", "so101"]
    agent: Union[SO100, SO101]

    def _setup_base(
        self,
        *,
        config: EnvironmentConfig,
        robot_uids: str,
        robot_cfgs: dict[str, dict],
        robot_color: tuple[float, float, float, float] | None,
        camera_mode: CameraMode,
        robot_init_qpos_noise: float,
    ) -> None:
        if robot_uids not in robot_cfgs:
            raise ValueError(f"robot_uids must be one of {list(robot_cfgs)}, got {robot_uids!r}")
        if camera_mode not in ("fixed", "wrist", "both"):
            raise ValueError(f"camera_mode must be fixed|wrist|both, got {camera_mode!r}")
        if config.camera.width <= 0 or config.camera.height <= 0:
            raise ValueError(
                f"camera dimensions must be > 0, got {config.camera.width}x{config.camera.height}"
            )

        self.config = config
        self.robot_color = robot_color
        self.camera_mode: CameraMode = camera_mode
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.camera_width = config.camera.width
        self.camera_height = config.camera.height
        self._robot_cfg = robot_cfgs[robot_uids]
        self._initial_obj_z: torch.Tensor | None = None

    @staticmethod
    def _reach_progress(dist: torch.Tensor) -> torch.Tensor:
        return 1.0 - torch.tanh(5.0 * dist)

    def _assemble_normalized_reward(
        self,
        *,
        reach_progress: torch.Tensor,
        is_grasped: torch.Tensor,
        task_progress: torch.Tensor,
        is_complete: torch.Tensor,
        action_delta_norm: float | torch.Tensor = 0.0,
    ) -> torch.Tensor:
        cfg = self.config.reward
        base = (
            cfg.reaching * reach_progress
            + cfg.grasping * is_grasped
            + cfg.task_objective * task_progress
            + cfg.completion_bonus * is_complete
        )
        return base - cfg.action_delta_penalty * action_delta_norm

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

    def _after_reconfigure(self, options: dict) -> None:
        self._initial_obj_z = torch.zeros(self.num_envs, device=self.device)

    def _build_ground(self) -> None:
        ground_builder = self.scene.create_actor_builder()
        ground_builder.add_plane_collision(
            sapien.Pose(p=[0, 0, 0], q=[0.7071068, 0, -0.7071068, 0])
        )
        ground_builder.initial_pose = sapien.Pose(p=[0, 0, 0])
        if self.scene.parallel_in_single_scene:
            ground_builder.set_scene_idxs([0])
        ground = ground_builder.build_static(name="ground")

        if not self.scene.can_render():
            return

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
        mat.base_color = self.config.ground_color
        shape = sapien.render.RenderShapeTriangleMesh(
            vertices=verts,
            triangles=tris,
            normals=normals,
            uvs=uvs,
            material=mat,
        )
        for obj in ground._objs:
            comp = sapien.render.RenderBodyComponent()
            comp.attach(shape)
            obj.add_component(comp)

    def _apply_robot_color_if_needed(self) -> None:
        if self.robot_color is None:
            return
        color = list(self.robot_color)
        for link in self.agent.robot.links:
            for obj in link._objs:
                render_body: RenderBodyComponent = obj.entity.find_component_by_type(
                    RenderBodyComponent
                )
                if render_body is None:
                    continue
                for render_shape in render_body.render_shapes:
                    for part in render_shape.parts:
                        part.material.set_base_color(color)

    def _reset_robot(self, env_idx: torch.Tensor) -> None:
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

    def _store_initial_obj_z(self, env_idx: torch.Tensor, z: torch.Tensor) -> None:
        if self._initial_obj_z is None:
            self._initial_obj_z = torch.zeros(self.num_envs, device=self.device)
        self._initial_obj_z[env_idx] = z

    def compute_normalized_dense_reward(
        self,
        obs,
        action: torch.Tensor,
        info: dict,
    ) -> torch.Tensor:
        return self.compute_dense_reward(obs=obs, action=action, info=info)
