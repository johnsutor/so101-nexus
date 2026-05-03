"""ManiSkill look-at environment."""

from __future__ import annotations

from typing import ClassVar

from so101_nexus_core.config import LookAtConfig
from so101_nexus_core.tasks import make_look_at_spec
from so101_nexus_maniskill.base_env import register_robot_variant
from so101_nexus_maniskill.primitive_target_env import PrimitiveTargetManiSkillEnv

_DEFAULT_CONFIG = LookAtConfig()


class LookAtEnv(PrimitiveTargetManiSkillEnv):
    """LookAt primitive: orient the TCP toward a sampled target object.

    The target is a static cube placed randomly in the workspace.
    """

    config: LookAtConfig
    default_config_cls: ClassVar[type[LookAtConfig]] = LookAtConfig

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
        super().__init__(
            spec=make_look_at_spec(config),
            config=config,
            *args,  # noqa: B026
            robot_uids=robot_uids,
            num_envs=num_envs,
            reconfiguration_freq=reconfiguration_freq,
            **kwargs,
        )


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
