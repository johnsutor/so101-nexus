"""Tests for the so101-nexus argparse CLI."""

from __future__ import annotations

import os
import sys

import pytest

from so101_nexus import cli as nexus_cli
from so101_nexus.testing.cli_contract import (
    run_main_dispatch_contract,
    run_parser_contract,
    run_parser_requires_subcommand,
)


def test_build_parser_has_shared_teleop_contract():
    run_parser_contract(nexus_cli)


def test_build_parser_requires_subcommand():
    run_parser_requires_subcommand(nexus_cli, pytest=pytest)


def test_main_dispatches_teleop(monkeypatch):
    """`main()` parses args and forwards to teleop_main with backend='mujoco'."""
    run_main_dispatch_contract(
        nexus_cli,
        backend="mujoco",
        argv0="so101-nexus",
        monkeypatch=monkeypatch,
    )


def test_main_sets_egl_for_teleop_when_gl_backend_is_unset(monkeypatch):
    """Teleop defaults to EGL so rgb_array rendering works headlessly on Linux."""
    called = {}

    def _fake_teleop_main(args, backend: str):
        called["backend"] = backend
        called["gl"] = os.environ.get("MUJOCO_GL")

    import so101_nexus.teleop.app as app_mod

    monkeypatch.setattr(app_mod, "main", _fake_teleop_main)
    monkeypatch.setattr("sys.argv", ["so101-nexus", "teleop"])
    monkeypatch.delenv("MUJOCO_GL", raising=False)

    nexus_cli.main()

    assert called["backend"] == "mujoco"
    assert called["gl"] == "egl"


def test_main_forces_mujoco_gl_egl_for_teleop(monkeypatch):
    """Teleop must force EGL even if the parent shell selected another backend."""
    called = {}

    def _fake_teleop_main(args, backend: str):
        called["backend"] = backend
        called["gl"] = os.environ.get("MUJOCO_GL")

    import so101_nexus.teleop.app as app_mod

    monkeypatch.setattr(app_mod, "main", _fake_teleop_main)
    monkeypatch.setattr(sys, "argv", ["so101-nexus", "teleop"])
    monkeypatch.setenv("MUJOCO_GL", "glfw")

    nexus_cli.main()

    assert called["backend"] == "mujoco"
    assert called["gl"] == "egl"


def test_main_warns_when_running_inside_vscode(monkeypatch, capsys):
    import so101_nexus.teleop.app as app_mod

    monkeypatch.setattr(app_mod, "main", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(sys, "argv", ["so101-nexus", "teleop"])
    monkeypatch.setenv("TERM_PROGRAM", "vscode")

    nexus_cli.main()

    captured = capsys.readouterr()
    assert "VS Code" in captured.err or "VS Code" in captured.out
