"""Command-line entry point for the so101-nexus-maniskill package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from so101_nexus_core.teleop.cli import build_teleop_parser, run_teleop

if TYPE_CHECKING:
    import argparse


def _build_parser() -> argparse.ArgumentParser:
    """Backwards-compatible alias used by tests."""
    return build_teleop_parser(prog="so101-nexus-maniskill")


def _setup() -> None:
    import so101_nexus_maniskill  # noqa: F401 — register gym envs eagerly


def main() -> None:
    """Parse args and dispatch teleop."""
    run_teleop("maniskill", pre_dispatch=_setup)
