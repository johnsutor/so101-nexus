"""ManiSkill directional move environment."""

from __future__ import annotations

from typing import ClassVar

from so101_nexus_core.config import MoveConfig
from so101_nexus_core.tasks import make_move_spec
from so101_nexus_maniskill.base_env import register_robot_variant
from so101_nexus_maniskill.primitive_target_env import PrimitiveTargetManiSkillEnv

_DEFAULT_CONFIG = MoveConfig()


class MoveEnv(PrimitiveTargetManiSkillEnv):
    """Move primitive: translate TCP a fixed distance in a specified direction.

    The target is a visual-only sphere. No graspable objects.
    """

    config: MoveConfig
    default_config_cls: ClassVar[type[MoveConfig]] = MoveConfig

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
        super().__init__(
            spec=make_move_spec(config),
            config=config,
            *args,  # noqa: B026
            robot_uids=robot_uids,
            num_envs=num_envs,
            reconfiguration_freq=reconfiguration_freq,
            **kwargs,
        )


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
