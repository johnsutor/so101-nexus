"""Command-line entry point for the so101-nexus-maniskill package."""

from __future__ import annotations

import argparse

from so101_nexus_core.teleop.leader import DEFAULT_WRIST_ROLL_OFFSET_DEG


def _build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argparse parser for the maniskill backend CLI."""
    parser = argparse.ArgumentParser(prog="so101-nexus-maniskill")
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


def main() -> None:
    """Dispatch to the requested subcommand."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "teleop":
        import so101_nexus_maniskill  # noqa: F401 — register gym envs eagerly
        from so101_nexus_core.teleop.app import main as teleop_main

        teleop_main(args)
    else:  # pragma: no cover
        parser.error(f"unknown command: {args.command}")
