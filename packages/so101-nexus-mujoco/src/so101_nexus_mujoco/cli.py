"""Command-line entry point for the so101-nexus-mujoco package."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from so101_nexus_core.teleop.cli import build_teleop_parser, run_teleop

if TYPE_CHECKING:
    import argparse


def _build_parser() -> argparse.ArgumentParser:
    """Backwards-compatible alias used by tests."""
    return build_teleop_parser(prog="so101-nexus-mujoco")


def _setup() -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    import so101_nexus_mujoco  # noqa: F401 — register gym envs eagerly


def main() -> None:
    """Parse args and dispatch teleop, configuring MuJoCo for headless render."""
    run_teleop("mujoco", pre_dispatch=_setup)
