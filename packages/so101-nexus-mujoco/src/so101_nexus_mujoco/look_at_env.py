"""Primitive look-at environment for SO-101."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from so101_nexus_core.config import ControlMode, LookAtConfig
from so101_nexus_core.tasks import make_look_at_spec
from so101_nexus_mujoco.primitive_target_env import PrimitiveTargetMuJoCoEnv

if TYPE_CHECKING:
    from so101_nexus_core.config import ObsMode


class LookAtEnv(PrimitiveTargetMuJoCoEnv):
    """LookAt primitive: orient the wrist camera toward a sampled target object.

    Default obs (6,): joint_positions.
    """

    config: LookAtConfig
    default_config_cls: ClassVar[type[LookAtConfig]] = LookAtConfig

    def __init__(
        self,
        config: LookAtConfig | None = None,
        render_mode: str | None = None,
        control_mode: ControlMode = "pd_joint_pos",
        robot_init_qpos_noise: float = 0.02,
        obs_mode: ObsMode = "state",
    ) -> None:
        if config is None:
            config = LookAtConfig()
        super().__init__(
            spec=make_look_at_spec(config),
            config=config,
            render_mode=render_mode,
            control_mode=control_mode,
            robot_init_qpos_noise=robot_init_qpos_noise,
        )
