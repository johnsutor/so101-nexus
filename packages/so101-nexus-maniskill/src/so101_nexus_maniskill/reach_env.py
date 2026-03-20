"""ManiSkill reach-to-target environment."""

from __future__ import annotations

from typing import Any

import sapien
import torch
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose

from so101_nexus_core.config import ReachConfig
from so101_nexus_core.observations import JointPositions
from so101_nexus_core.robot_presets import build_maniskill_robot_configs
from so101_nexus_maniskill.base_env import SO101NexusManiSkillBaseEnv

_DEFAULT_CONFIG = ReachConfig()


class ReachEnv(SO101NexusManiSkillBaseEnv):
    """Reach primitive: move TCP to a randomly sampled 3-D target position.

    The target is a visual-only sphere. No graspable objects.
    """

    config: ReachConfig
    task_description = "Move the robot's end-effector to the target position."

    def __init__(
        self,
        *args,
        config: ReachConfig = ReachConfig(),
        robot_uids: str = "so100",
        num_envs: int = 1,
        reconfiguration_freq: int | None = None,
        **kwargs,
    ):
        if config.observations is None:
            config.observations = [JointPositions()]

        robot_cfgs = build_maniskill_robot_configs(config=config)
        self._setup_base(config=config, robot_uids=robot_uids, robot_cfgs=robot_cfgs)

        if reconfiguration_freq is None:
            reconfiguration_freq = 1 if config.camera_mode in ("wrist", "both") else 0

        self._target_pos: torch.Tensor | None = None

        super().__init__(
            *args,
            robot_uids=robot_uids,
            reconfiguration_freq=reconfiguration_freq,
            num_envs=num_envs,
            **kwargs,
        )

    def _load_scene(self, options: dict) -> None:
        self._build_ground()
        targets: list[Actor] = []
        for i in range(self.num_envs):
            builder = self.scene.create_actor_builder()
            builder.add_sphere_visual(
                radius=self.config.target_radius,
                material=sapien.render.RenderMaterial(
                    base_color=[1.0, 0.5, 0.0, 0.7],
                ),
            )
            builder.initial_pose = sapien.Pose(p=[0.15, 0, 0.15])
            builder.set_scene_idxs([i])
            target = builder.build_kinematic(name=f"reach_target-{i}")
            targets.append(target)
            self.remove_from_state_dict_registry(target)
        self.target_site = Actor.merge(targets, name="reach_target")
        self.add_to_state_dict_registry(self.target_site)
        self._apply_robot_color_if_needed()

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict) -> None:
        with torch.device(self.device):
            b = len(env_idx)
            self._reset_robot(env_idx)

            half = self.config.target_workspace_half_extent
            center = torch.tensor([0.15, 0.0, 0.15], device=self.device)
            offset = (torch.rand(b, 3, device=self.device) * 2 - 1) * half
            pos = center.unsqueeze(0) + offset
            pos[:, 2] = pos[:, 2].clamp(min=0.05)
            q = torch.tensor(
                [[1, 0, 0, 0]], device=self.device, dtype=torch.float32
            ).expand(b, -1)
            self.target_site.set_pose(Pose.create_from_pq(p=pos, q=q))
            self._target_pos = pos

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
        from so101_nexus_core.observations import TargetOffset

        if isinstance(component, TargetOffset):
            obs["target_offset"] = self.target_site.pose.p - self.agent.tcp_pose.p
        else:
            super()._add_component_obs(obs, component, info)

    def compute_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ) -> torch.Tensor:
        """Tanh-shaped reach reward with completion bonus."""
        reach = self._reach_progress(info["tcp_to_target_dist"])
        bonus = self.config.reward.completion_bonus
        return (1.0 - bonus) * reach + bonus * info["success"]


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


ReachSO100Env = _register_robot_variant(
    class_name="ReachSO100Env",
    env_id="ManiSkillReachSO100-v1",
    base_cls=ReachEnv,
    robot_uid="so100",
)
ReachSO101Env = _register_robot_variant(
    class_name="ReachSO101Env",
    env_id="ManiSkillReachSO101-v1",
    base_cls=ReachEnv,
    robot_uid="so101",
)
