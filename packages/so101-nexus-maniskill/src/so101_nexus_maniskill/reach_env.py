"""ManiSkill reach-to-target environment."""

from __future__ import annotations

from typing import ClassVar

from so101_nexus_core.config import ReachConfig
from so101_nexus_core.tasks import make_reach_spec
from so101_nexus_maniskill.base_env import register_robot_variant
from so101_nexus_maniskill.primitive_target_env import PrimitiveTargetManiSkillEnv

_DEFAULT_CONFIG = ReachConfig()


class ReachEnv(PrimitiveTargetManiSkillEnv):
    """Reach primitive: move TCP to a randomly sampled 3-D target position.

    The target is a visual-only sphere. No graspable objects.
    """

    config: ReachConfig
    default_config_cls: ClassVar[type[ReachConfig]] = ReachConfig

    def __init__(
        self,
        *args,
        config: ReachConfig | None = None,
        robot_uids: str = "so100",
        num_envs: int = 1,
        reconfiguration_freq: int | None = None,
        **kwargs,
    ):
        if config is None:
            config = ReachConfig()
        super().__init__(
            spec=make_reach_spec(config),
            config=config,
            *args,  # noqa: B026
            robot_uids=robot_uids,
            num_envs=num_envs,
            reconfiguration_freq=reconfiguration_freq,
            **kwargs,
        )


ReachSO100Env = register_robot_variant(
    class_name="ReachSO100Env",
    env_id="ManiSkillReachSO100-v1",
    base_cls=ReachEnv,
    robot_uid="so100",
    max_episode_steps=_DEFAULT_CONFIG.max_episode_steps,
    caller_globals=globals(),
)
ReachSO101Env = register_robot_variant(
    class_name="ReachSO101Env",
    env_id="ManiSkillReachSO101-v1",
    base_cls=ReachEnv,
    robot_uid="so101",
    max_episode_steps=_DEFAULT_CONFIG.max_episode_steps,
    caller_globals=globals(),
)
