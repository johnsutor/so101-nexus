"""Gradio-based teleop recorder for SO101-Nexus environments.

This subpackage ships with ``so101-nexus-core`` and provides the shared
teleoperation recorder used by the MuJoCo and ManiSkill backends. All heavy
dependencies (``gradio``, ``lerobot``, ``plotly``, ``cv2``) are imported
lazily inside functions so that importing ``so101_nexus_core.teleop`` works
on a base install without the optional ``teleop`` extra.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse

__all__ = ["main", "run_app"]


def main(args: argparse.Namespace | None = None) -> None:  # pragma: no cover - thin wrapper
    """Launch the Gradio teleop recorder. Entry point for CLI subcommands."""
    from so101_nexus_core.teleop.app import main as _main

    _main(args)


def run_app(args: argparse.Namespace | None = None) -> None:  # pragma: no cover - alias
    """Alias for :func:`main` kept for explicit callers."""
    main(args)
