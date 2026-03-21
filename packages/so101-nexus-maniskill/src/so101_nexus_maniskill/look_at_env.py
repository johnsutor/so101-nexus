"""ManiSkill look-at environment."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from mani_skill.utils.building import actors
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose

from so101_nexus_core.config import LookAtConfig
from so101_nexus_core.constants import sample_color
from so101_nexus_core.observations import GazeDirection

if TYPE_CHECKING:
    from so101_nexus_core.objects import CubeObject
from so101_nexus_core.robot_presets import build_maniskill_robot_configs
from so101_nexus_maniskill.base_env import SO101NexusManiSkillBaseEnv, register_robot_variant

_DEFAULT_CONFIG = LookAtConfig()


class LookAtEnv(SO101NexusManiSkillBaseEnv):
    """LookAt primitive: orient the TCP toward a sampled target object.

    The target is a static cube placed randomly in the workspace.
    """

    config: LookAtConfig

    def __init__(
        self,
        *args,
        config: LookAtConfig | None = None,
        robot_uids: str = "so100",
        num_envs: int = 1,
        reconfiguration_freq: int | None = None,
        **kwargs,
    ):
        if config is None:
            config = LookAtConfig()
        self._target_obj: CubeObject = config.objects[0]  # type: ignore[assignment]

        robot_cfgs = build_maniskill_robot_configs(config=config)
        self._setup_base(config=config, robot_uids=robot_uids, robot_cfgs=robot_cfgs)

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
        """Return the current episode task description."""
        return f"Look at the {self._target_obj!r}."

    def _load_scene(self, options: dict) -> None:
        self._build_ground()
        obj = self._target_obj
        targets: list[Actor] = []
        for i in range(self.num_envs):
            target = actors.build_cube(
                self.scene,
                half_size=obj.half_size,
                color=sample_color(obj.color),
                name=f"look_target-{i}",
                body_type="kinematic",
                scene_idxs=[i],
            )
            targets.append(target)
            self.remove_from_state_dict_registry(target)
        self.target_obj_actor = Actor.merge(targets, name="look_target")
        self.add_to_state_dict_registry(self.target_obj_actor)
        self._apply_robot_color_if_needed()

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict) -> None:
        with torch.device(self.device):
            b = len(env_idx)
            self._reset_robot(env_idx)

            half = self.config.spawn_half_size
            cx, cy = self.config.spawn_center
            x = cx + (torch.rand(b, device=self.device) * 2 - 1) * half
            y = cy + (torch.rand(b, device=self.device) * 2 - 1) * half
            z = torch.full((b,), self._target_obj.half_size, device=self.device)
            pos = torch.stack([x, y, z], dim=1)
            q = torch.tensor([[1, 0, 0, 0]], device=self.device, dtype=torch.float32).expand(b, -1)
            self.target_obj_actor.set_pose(Pose.create_from_pq(p=pos, q=q))

    def evaluate(self) -> dict[str, torch.Tensor]:
        """Return per-env metrics: orientation_error, success."""
        tcp_pose = self.agent.tcp_pose
        # TCP forward = z-axis of rotation matrix (third column)
        rot_mat = tcp_pose.to_transformation_matrix()[..., :3, :3]
        tcp_forward = rot_mat[..., :, 2]  # (num_envs, 3)
        to_target = self.target_obj_actor.pose.p - tcp_pose.p
        to_target_norm = to_target / (torch.linalg.norm(to_target, dim=1, keepdim=True) + 1e-8)
        cos_sim = (tcp_forward * to_target_norm).sum(dim=1).clamp(-1, 1)
        orientation_error = torch.arccos(cos_sim)
        return {
            "orientation_error": orientation_error,
            "success": orientation_error < self.config._orientation_success_threshold_rad,
        }

    def _get_obs_extra(self, info: dict) -> dict[str, torch.Tensor]:
        return self._build_obs_extra_from_components(info)

    def _add_component_obs(
        self, obs: dict[str, torch.Tensor], component: object, info: dict
    ) -> None:
        if isinstance(component, GazeDirection):
            to_target = self.target_obj_actor.pose.p - self.agent.tcp_pose.p
            obs["gaze_direction"] = to_target / (
                torch.linalg.norm(to_target, dim=1, keepdim=True) + 1e-8
            )
        else:
            super()._add_component_obs(obs, component, info)

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict) -> torch.Tensor:
        """Cosine similarity reward for orientation toward target."""
        tcp_pose = self.agent.tcp_pose
        rot_mat = tcp_pose.to_transformation_matrix()[..., :3, :3]
        tcp_forward = rot_mat[..., :, 2]
        to_target = self.target_obj_actor.pose.p - tcp_pose.p
        to_target_norm = to_target / (torch.linalg.norm(to_target, dim=1, keepdim=True) + 1e-8)
        cos_sim = (tcp_forward * to_target_norm).sum(dim=1).clamp(-1, 1)
        orient = (cos_sim + 1) / 2  # map [-1, 1] to [0, 1]
        bonus = self.config.reward.completion_bonus
        return (1.0 - bonus) * orient + bonus * info["success"]


LookAtSO100Env = register_robot_variant(
    class_name="LookAtSO100Env",
    env_id="ManiSkillLookAtSO100-v1",
    base_cls=LookAtEnv,
    robot_uid="so100",
    max_episode_steps=_DEFAULT_CONFIG.max_episode_steps,
    caller_globals=globals(),
)
LookAtSO101Env = register_robot_variant(
    class_name="LookAtSO101Env",
    env_id="ManiSkillLookAtSO101-v1",
    base_cls=LookAtEnv,
    robot_uid="so101",
    max_episode_steps=_DEFAULT_CONFIG.max_episode_steps,
    caller_globals=globals(),
)
