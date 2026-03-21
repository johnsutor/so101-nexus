"""ManiSkill directional move environment."""

from __future__ import annotations

from typing import Any

import sapien
import torch
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose

from so101_nexus_core.config import DIRECTION_VECTORS, MoveConfig
from so101_nexus_core.observations import TargetOffset
from so101_nexus_core.robot_presets import build_maniskill_robot_configs
from so101_nexus_maniskill.base_env import SO101NexusManiSkillBaseEnv, register_robot_variant

_DEFAULT_CONFIG = MoveConfig()


class MoveEnv(SO101NexusManiSkillBaseEnv):
    """Move primitive: translate TCP a fixed distance in a specified direction.

    The target is a visual-only sphere. No graspable objects.
    """

    config: MoveConfig

    def __init__(
        self,
        *args,
        config: MoveConfig | None = None,
        robot_uids: str = "so100",
        num_envs: int = 1,
        reconfiguration_freq: int | None = None,
        **kwargs,
    ):
        if config is None:
            config = MoveConfig()
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
    def task_description(self) -> str:
        """Return a description of the current move task."""
        return (
            f"Move the end-effector {self.config.direction} by {self.config.target_distance:.2f} m."
        )

    def _load_scene(self, options: dict) -> None:
        self._build_ground()
        targets: list[Actor] = []
        for i in range(self.num_envs):
            builder = self.scene.create_actor_builder()
            builder.add_sphere_visual(
                radius=0.015,
                material=sapien.render.RenderMaterial(
                    base_color=[0.0, 0.8, 0.2, 0.7],
                ),
            )
            builder.initial_pose = sapien.Pose(p=[0.15, 0, 0.1])
            builder.set_scene_idxs([i])
            target = builder.build_kinematic(name=f"move_target-{i}")
            targets.append(target)
            self.remove_from_state_dict_registry(target)
        self.target_site = Actor.merge(targets, name="move_target")
        self.add_to_state_dict_registry(self.target_site)
        self._apply_robot_color_if_needed()

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict) -> None:
        with torch.device(self.device):
            b = len(env_idx)
            self._reset_robot(env_idx)

            # Step physics once to get accurate TCP position after robot reset
            self.scene.step()

            tcp_pos = self.agent.tcp_pose.p.clone()
            dir_vec = torch.tensor(
                DIRECTION_VECTORS[self.config.direction],
                device=self.device,
                dtype=torch.float32,
            )
            target = tcp_pos + dir_vec * self.config.target_distance
            target[:, 2] = target[:, 2].clamp(min=0.02)
            q = torch.tensor([[1, 0, 0, 0]], device=self.device, dtype=torch.float32).expand(b, -1)
            self.target_site.set_pose(Pose.create_from_pq(p=target, q=q))
            self._target_pos = target

    def evaluate(self) -> dict[str, torch.Tensor]:
        """Return per-env metrics: tcp_to_target_dist, success."""
        tcp_to_target = self.target_site.pose.p - self.agent.tcp_pose.p
        dist = torch.linalg.norm(tcp_to_target, axis=1)
        return {
            "tcp_to_target_dist": dist,
            "success": dist < self.config.success_threshold,
        }

    def _get_obs_extra(self, info: dict) -> dict[str, torch.Tensor]:
        return self._build_obs_extra_from_components(info)

    def _add_component_obs(
        self, obs: dict[str, torch.Tensor], component: object, info: dict
    ) -> None:
        if isinstance(component, TargetOffset):
            obs["target_offset"] = self.target_site.pose.p - self.agent.tcp_pose.p
        else:
            super()._add_component_obs(obs, component, info)

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict) -> torch.Tensor:
        """Tanh-shaped reach reward with completion bonus."""
        reach = self._reach_progress(info["tcp_to_target_dist"])
        bonus = self.config.reward.completion_bonus
        return (1.0 - bonus) * reach + bonus * info["success"]


MoveSO100Env = register_robot_variant(
    class_name="MoveSO100Env",
    env_id="ManiSkillMoveSO100-v1",
    base_cls=MoveEnv,
    robot_uid="so100",
    max_episode_steps=_DEFAULT_CONFIG.max_episode_steps,
    caller_globals=globals(),
)
MoveSO101Env = register_robot_variant(
    class_name="MoveSO101Env",
    env_id="ManiSkillMoveSO101-v1",
    base_cls=MoveEnv,
    robot_uid="so101",
    max_episode_steps=_DEFAULT_CONFIG.max_episode_steps,
    caller_globals=globals(),
)
