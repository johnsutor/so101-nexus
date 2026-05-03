"""Shared backend-agnostic CLI helpers for the teleop entry points.

Backend packages provide thin `main()` wrappers that supply (a) a backend
identifier and (b) an optional `pre_dispatch` callable for backend-specific
environment setup (e.g. setting `MUJOCO_GL`, eagerly importing the backend
package so its gym envs register).
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from so101_nexus_core.teleop.leader import DEFAULT_WRIST_ROLL_OFFSET_DEG

if TYPE_CHECKING:
    from collections.abc import Callable

    from so101_nexus_core.env_ids import Backend


def build_teleop_parser(prog: str) -> argparse.ArgumentParser:
    """Construct the shared argparse parser used by every backend CLI."""
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
    return parser


def run_teleop(
    backend: Backend,
    *,
    pre_dispatch: Callable[[], None] | None = None,
) -> None:
    """Parse argv, run any backend-specific setup, then invoke teleop.app.main."""
    parser = build_teleop_parser(prog=f"so101-nexus-{backend}")
    args = parser.parse_args()

    if args.command == "teleop":
        if pre_dispatch is not None:
            pre_dispatch()
        from so101_nexus_core.teleop.app import main as teleop_main

        teleop_main(args, backend=backend)
    else:  # pragma: no cover - argparse enforces valid commands
        parser.error(f"unknown command: {args.command}")
