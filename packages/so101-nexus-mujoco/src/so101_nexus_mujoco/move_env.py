"""Primitive directional move environment for SO-101."""

from __future__ import annotations

from typing import ClassVar

from so101_nexus_core.config import ControlMode, MoveConfig
from so101_nexus_core.tasks import make_move_spec
from so101_nexus_mujoco.primitive_target_env import PrimitiveTargetMuJoCoEnv


class MoveEnv(PrimitiveTargetMuJoCoEnv):
    """Move primitive: translate TCP a fixed distance in a specified direction.

    Default obs (6,): joint_positions.
    """

    config: MoveConfig
    default_config_cls: ClassVar[type[MoveConfig]] = MoveConfig

    def __init__(
        self,
        config: MoveConfig | None = None,
        render_mode: str | None = None,
        control_mode: ControlMode = "pd_joint_pos",
        robot_init_qpos_noise: float = 0.02,
    ) -> None:
        if config is None:
            config = MoveConfig()
        super().__init__(
            spec=make_move_spec(config),
            config=config,
            render_mode=render_mode,
            control_mode=control_mode,
            robot_init_qpos_noise=robot_init_qpos_noise,
        )
