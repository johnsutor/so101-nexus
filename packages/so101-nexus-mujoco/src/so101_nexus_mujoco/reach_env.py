"""Primitive reach-to-target environment for SO-101."""

from __future__ import annotations

from typing import ClassVar

from so101_nexus_core.config import ControlMode, ReachConfig
from so101_nexus_core.tasks import make_reach_spec
from so101_nexus_mujoco.primitive_target_env import PrimitiveTargetMuJoCoEnv


class ReachEnv(PrimitiveTargetMuJoCoEnv):
    """Reach primitive: move TCP to a randomly sampled 3-D target site.

    Default obs (6,): joint_positions.
    """

    config: ReachConfig
    default_config_cls: ClassVar[type[ReachConfig]] = ReachConfig

    def __init__(
        self,
        config: ReachConfig | None = None,
        render_mode: str | None = None,
        control_mode: ControlMode = "pd_joint_pos",
        robot_init_qpos_noise: float = 0.02,
    ) -> None:
        if config is None:
            config = ReachConfig()
        super().__init__(
            spec=make_reach_spec(config),
            config=config,
            render_mode=render_mode,
            control_mode=control_mode,
            robot_init_qpos_noise=robot_init_qpos_noise,
        )
