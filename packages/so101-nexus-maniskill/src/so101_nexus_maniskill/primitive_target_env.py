"""ManiSkill backend for the PrimitiveTargetSpec abstraction."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import sapien
import torch
from mani_skill.utils.building import actors
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose

from so101_nexus_core.constants import sample_color
from so101_nexus_core.robot_presets import build_maniskill_robot_configs
from so101_nexus_core.tasks import PrimitiveTargetSpec, TorchContext, resolve_task_description
from so101_nexus_maniskill.base_env import SO101NexusManiSkillBaseEnv

if TYPE_CHECKING:
    from so101_nexus_core.config import EnvironmentConfig


class PrimitiveTargetManiSkillEnv(SO101NexusManiSkillBaseEnv):
    """Generic ManiSkill env consuming a PrimitiveTargetSpec."""

    def __init__(
        self,
        spec: PrimitiveTargetSpec,
        config: EnvironmentConfig,
        *args,
        robot_uids: str = "so100",
        num_envs: int = 1,
        reconfiguration_freq: int | None = None,
        **kwargs,
    ):
        self._spec = spec
        self._task_description = resolve_task_description(spec, config)
        robot_cfgs = build_maniskill_robot_configs(config=config)
        self._setup_base(config=config, robot_uids=robot_uids, robot_cfgs=robot_cfgs)
        self._target_pos: torch.Tensor | None = None
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
    def task_description(self) -> str:  # noqa: D102
        return self._task_description

    def _build_kinematic_marker(self, name: str, env_idx: int) -> Actor:
        marker = self._spec.marker
        builder = self.scene.create_actor_builder()
        if marker.shape == "sphere":
            builder.add_sphere_visual(
                radius=marker.size,
                material=sapien.render.RenderMaterial(base_color=list(marker.rgba)),
            )
        else:
            builder.add_box_visual(
                half_size=[marker.size] * 3,
                material=sapien.render.RenderMaterial(base_color=list(marker.rgba)),
            )
        builder.initial_pose = sapien.Pose(p=[0.15, 0, 0.15])
        builder.set_scene_idxs([env_idx])
        return builder.build_kinematic(name=f"{name}-{env_idx}")

    def _build_dynamic_marker(self, name: str, env_idx: int) -> Actor:
        marker = self._spec.marker
        return actors.build_cube(
            self.scene,
            half_size=marker.size,
            color=sample_color(marker.color_name),
            name=f"{name}-{env_idx}",
            body_type="kinematic",
            scene_idxs=[env_idx],
        )

    def _load_scene(self, options: dict) -> None:
        self._build_ground()
        targets: list[Actor] = []
        for i in range(self.num_envs):
            if self._spec.marker.is_kinematic:
                target = self._build_kinematic_marker(self._spec.marker.name, i)
            else:
                target = self._build_dynamic_marker(self._spec.marker.name, i)
            targets.append(target)
            self.remove_from_state_dict_registry(target)
        self.target_site = Actor.merge(targets, name=self._spec.marker.name)
        self.add_to_state_dict_registry(self.target_site)
        self._apply_robot_color_if_needed()

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict) -> None:
        with torch.device(self.device):
            b = len(env_idx)
            self._reset_robot(env_idx)

            tcp_pos = None
            if self._spec.requires_tcp_pos_for_sampling:
                self.scene.step()
                tcp_pos = self.agent.tcp_pose.p.clone()

            ctx = TorchContext(
                rng=None,
                device=self.device,
                config=self.config,
                batch_size=b,
                tcp_pos=tcp_pos,
            )
            pos = self._spec.sampler.sample_torch(ctx)
            q = torch.tensor([[1, 0, 0, 0]], device=self.device, dtype=torch.float32).expand(b, -1)
            self.target_site.set_pose(Pose.create_from_pq(p=pos, q=q))
            self._target_pos = pos

    def evaluate(self) -> dict[str, torch.Tensor]:  # noqa: D102
        spec = self._spec
        target_pos = self.target_site.pose.p
        tcp_pose = self.agent.tcp_pose
        tcp_pos = tcp_pose.p
        tcp_forward = None
        if spec.requires_tcp_forward_for_metric:
            rot_mat = tcp_pose.to_transformation_matrix()[..., :3, :3]
            tcp_forward = rot_mat[..., :, 2]
        ctx = TorchContext(
            rng=None,
            device=self.device,
            config=self.config,
            batch_size=tcp_pos.shape[0],
        )
        metric, success = spec.metric.evaluate_torch(
            target_pos=target_pos,
            tcp_pos=tcp_pos,
            tcp_forward=tcp_forward,
            ctx=ctx,
        )
        metric_key = (
            "orientation_error" if spec.requires_tcp_forward_for_metric else "tcp_to_target_dist"
        )
        return {metric_key: metric, "success": success}

    def _get_obs_extra(self, info: dict) -> dict[str, torch.Tensor]:
        return self._build_obs_extra_from_components(info)

    def _add_component_obs(self, obs, component, info) -> None:
        from so101_nexus_core.observations import GazeDirection, TargetOffset

        if isinstance(component, TargetOffset):
            obs["target_offset"] = self.target_site.pose.p - self.agent.tcp_pose.p
        elif isinstance(component, GazeDirection):
            to_target = self.target_site.pose.p - self.agent.tcp_pose.p
            obs["gaze_direction"] = to_target / (
                torch.linalg.norm(to_target, dim=1, keepdim=True) + 1e-8
            )
        else:
            super()._add_component_obs(obs, component, info)

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict) -> torch.Tensor:  # noqa: D102
        spec = self._spec
        ctx = TorchContext(
            rng=None,
            device=self.device,
            config=self.config,
            batch_size=action.shape[0],
        )
        metric_key = (
            "orientation_error" if spec.requires_tcp_forward_for_metric else "tcp_to_target_dist"
        )
        progress = spec.shaper.shape_torch(info[metric_key], ctx)
        completion = self.config.reward.completion_bonus
        return (1.0 - completion) * progress + completion * info["success"].to(progress.dtype)
