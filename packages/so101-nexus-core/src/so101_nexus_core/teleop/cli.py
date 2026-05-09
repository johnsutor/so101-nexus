"""Shared CLI helpers for backend teleop entry points."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from so101_nexus_core.teleop.leader import DEFAULT_WRIST_ROLL_OFFSET_DEG

if TYPE_CHECKING:
    from collections.abc import Callable

    from so101_nexus_core.env_ids import Backend


def build_teleop_parser(prog: str) -> argparse.ArgumentParser:
    """Construct the shared parser used by backend CLI modules."""
    parser = argparse.ArgumentParser(prog=prog)
    sub = parser.add_subparsers(dest="command", required=True)

    teleop = sub.add_parser("teleop", help="Launch the Gradio teleop recorder")
    teleop.add_argument("--leader-port", type=str, default="/dev/ttyACM0")
    teleop.add_argument("--leader-id", type=str, default="so101_leader")
    teleop.add_argument(
        "--wrist-roll-offset-deg",
        type=float,
        default=DEFAULT_WRIST_ROLL_OFFSET_DEG,
    )
    teleop.add_argument("--env-config-profile", type=str, default=None)
    teleop.add_argument("--env-config-factory", type=str, default=None)
    teleop.add_argument("--env-module", action="append", default=[], dest="env_modules")
    teleop.add_argument("--extra-env-id", action="append", default=[], dest="extra_env_ids")
    return parser


def run_teleop(
    backend: Backend,
    *,
    pre_dispatch: Callable[[], None] | None = None,
) -> None:
    """Parse argv, run backend setup, and launch the shared teleop app."""
    parser = build_teleop_parser(prog=f"so101-nexus-{backend}")
    args = parser.parse_args()

    if args.command == "teleop":
        if pre_dispatch is not None:
            pre_dispatch()
        from so101_nexus_core.teleop.app import main as teleop_main

        teleop_main(args, backend=backend)
    else:  # pragma: no cover - argparse enforces valid commands
        parser.error(f"unknown command: {args.command}")
