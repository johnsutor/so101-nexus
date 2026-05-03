"""Tests for the mujoco-backend argparse CLI.

Most assertions delegate to the shared `cli_contract` helpers; the two
`MUJOCO_GL` environment-variable tests stay backend-local because they
exercise a MuJoCo-specific concern.
"""

from __future__ import annotations

import os

from so101_nexus_core.testing.cli_contract import (
    run_main_dispatches_teleop,
    run_parser_contract,
    run_parser_requires_subcommand,
    run_parser_wrist_roll_offset_contract,
)
from so101_nexus_mujoco import cli as mujoco_cli


def test_build_parser_has_teleop_subcommand():
    run_parser_contract(mujoco_cli)


def test_build_parser_wrist_roll_offset_parses():
    run_parser_wrist_roll_offset_contract(mujoco_cli)


def test_build_parser_requires_subcommand():
    run_parser_requires_subcommand(mujoco_cli)


def test_main_dispatches_teleop(monkeypatch):
    run_main_dispatches_teleop(
        mujoco_cli, backend="mujoco", monkeypatch=monkeypatch, argv_prog="so101-nexus-mujoco"
    )


def test_main_sets_egl_for_teleop_when_gl_backend_is_unset(monkeypatch):
    """Teleop defaults to EGL so rgb_array rendering works headlessly on Linux."""
    captured: dict = {}

    def _fake_app_main(args, backend: str):
        captured["gl"] = os.environ.get("MUJOCO_GL")

    import so101_nexus_core.teleop.app as app_mod

    monkeypatch.setattr(app_mod, "main", _fake_app_main)
    monkeypatch.setattr("sys.argv", ["so101-nexus-mujoco", "teleop"])
    monkeypatch.delenv("MUJOCO_GL", raising=False)

    mujoco_cli.main()

    assert captured["gl"] == "egl"


def test_main_preserves_explicit_mujoco_gl_override(monkeypatch):
    """Teleop must not overwrite an explicit backend selected by the user."""
    captured: dict = {}

    def _fake_app_main(args, backend: str):
        captured["gl"] = os.environ.get("MUJOCO_GL")

    import so101_nexus_core.teleop.app as app_mod

    monkeypatch.setattr(app_mod, "main", _fake_app_main)
    monkeypatch.setattr("sys.argv", ["so101-nexus-mujoco", "teleop"])
    monkeypatch.setenv("MUJOCO_GL", "osmesa")

    mujoco_cli.main()

    assert captured["gl"] == "osmesa"
