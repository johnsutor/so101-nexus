"""Command-line entry point for the so101-nexus library."""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

from so101_nexus.teleop.cli import parse_teleop_args, run_teleop

if TYPE_CHECKING:
    from so101_nexus.teleop.cli import TeleopArgs


def parse_args(argv: list[str] | None = None) -> TeleopArgs:
    """Parse the so101-nexus CLI arguments (exposed for the CLI contract tests)."""
    return parse_teleop_args(argv, prog="so101-nexus")


def _setup_teleop_backend() -> None:
    """Prepare MuJoCo for teleop recording before launching the app."""
    os.environ["MUJOCO_GL"] = "egl"
    if os.environ.get("TERM_PROGRAM") == "vscode":
        print(
            "Note: detected VS Code integrated terminal. Forcing MUJOCO_GL=egl "
            "to avoid GLFW/libdecor launch failures.",
            file=sys.stderr,
        )
    import so101_nexus.mujoco  # noqa: F401 - register gym envs eagerly


def main() -> None:
    """Dispatch to the requested subcommand."""
    run_teleop("mujoco", pre_dispatch=_setup_teleop_backend)
