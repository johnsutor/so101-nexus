"""Command-line entry point for the so101-nexus-maniskill package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from so101_nexus_core.teleop.cli import build_teleop_parser, run_teleop

if TYPE_CHECKING:
    import argparse


def _build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argparse parser for the maniskill backend CLI."""
    return build_teleop_parser(prog="so101-nexus-maniskill")


def _setup_teleop_backend() -> None:
    """Register ManiSkill gym envs before launching teleop."""
    import so101_nexus_maniskill  # noqa: F401 — register gym envs eagerly


def main() -> None:
    """Dispatch to the requested subcommand."""
    run_teleop("maniskill", pre_dispatch=_setup_teleop_backend)
