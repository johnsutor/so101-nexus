"""Shared CLI helpers for backend teleop entry points."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated

import tyro

from so101_nexus.teleop.leader import DEFAULT_WRIST_ROLL_OFFSET_DEG

if TYPE_CHECKING:
    from collections.abc import Callable

    from so101_nexus.env_ids import Backend


@dataclass
class TeleopArgs:
    """Parsed arguments for the ``teleop`` subcommand."""

    leader_port: str = "/dev/ttyACM0"
    leader_id: str = "so101_leader"
    wrist_roll_offset_deg: float = DEFAULT_WRIST_ROLL_OFFSET_DEG
    env_config_profile: str | None = None
    env_config_factory: str | None = None
    env_modules: Annotated[
        list[str], tyro.conf.arg(name="env-module"), tyro.conf.UseAppendAction
    ] = field(default_factory=list)
    extra_env_ids: Annotated[
        list[str], tyro.conf.arg(name="extra-env-id"), tyro.conf.UseAppendAction
    ] = field(default_factory=list)


def parse_teleop_args(argv: list[str] | None = None, *, prog: str = "so101-nexus") -> TeleopArgs:
    """Parse the ``teleop`` subcommand from *argv* (defaults to ``sys.argv``)."""
    return tyro.extras.subcommand_cli_from_dict({"teleop": TeleopArgs}, args=argv, prog=prog)


def run_teleop(
    backend: Backend,
    *,
    pre_dispatch: Callable[[], None] | None = None,
) -> None:
    """Parse argv, run backend setup, and launch the shared teleop app."""
    args = parse_teleop_args()
    if pre_dispatch is not None:
        pre_dispatch()
    from so101_nexus.teleop.app import main as teleop_main

    teleop_main(args, backend=backend)
