"""Command-line entry point for the so101-nexus-maniskill package."""

from __future__ import annotations

import argparse

_TELEOP_INCOMPATIBLE_HINT = (
    "Teleop cannot run in the same environment as so101-nexus-maniskill: "
    "lerobot[feetech]>=0.5.0 requires gymnasium>=1.1.1, but mani-skill pins "
    "gymnasium<1.1.1.\n\n"
    "Use the mujoco backend for teleop instead:\n"
    "  uv sync --package so101-nexus-mujoco --extra teleop\n"
    "  uv run so101-nexus-mujoco teleop --leader-port /dev/ttyACM0\n\n"
    "ManiSkill envs can still be referenced from the teleop UI when launched "
    "from the mujoco env, since the env id is resolved dynamically."
)


def _build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argparse parser for the maniskill backend CLI."""
    parser = argparse.ArgumentParser(prog="so101-nexus-maniskill")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser(
        "teleop",
        help="(unavailable) teleop and maniskill cannot share an environment",
    )
    return parser


def main() -> None:
    """Dispatch to the requested subcommand."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "teleop":
        raise SystemExit(_TELEOP_INCOMPATIBLE_HINT)
    parser.error(f"unknown command: {args.command}")  # pragma: no cover
