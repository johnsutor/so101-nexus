"""Command-line entry point for the so101-nexus-mujoco package."""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

from so101_nexus_core.teleop.cli import build_teleop_parser, run_teleop

if TYPE_CHECKING:
    import argparse


def _build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argparse parser for the mujoco backend CLI."""
    return build_teleop_parser(prog="so101-nexus-mujoco")


def _setup_teleop_backend() -> None:
    """Prepare MuJoCo for teleop recording before launching the app."""
    os.environ["MUJOCO_GL"] = "egl"
    if os.environ.get("TERM_PROGRAM") == "vscode":
        print(
            "Note: detected VS Code integrated terminal. Forcing MUJOCO_GL=egl "
            "to avoid GLFW/libdecor launch failures.",
            file=sys.stderr,
        )
    import so101_nexus_mujoco  # noqa: F401 — register gym envs eagerly


def main() -> None:
    """Dispatch to the requested subcommand."""
    run_teleop("mujoco", pre_dispatch=_setup_teleop_backend)
