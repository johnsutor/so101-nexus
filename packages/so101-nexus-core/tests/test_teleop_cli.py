"""Tests for the shared teleop CLI helpers in core."""

from __future__ import annotations

import pytest

from so101_nexus_core.teleop.cli import build_teleop_parser, run_teleop


def test_build_teleop_parser_accepts_teleop_subcommand():
    parser = build_teleop_parser(prog="so101-nexus-test")
    args = parser.parse_args(["teleop", "--leader-port", "/dev/null"])
    assert args.command == "teleop"
    assert args.leader_port == "/dev/null"
    assert args.leader_id == "so101_leader"


def test_build_teleop_parser_requires_subcommand():
    parser = build_teleop_parser(prog="so101-nexus-test")
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_build_teleop_parser_wrist_roll_offset_parses():
    parser = build_teleop_parser(prog="so101-nexus-test")
    args = parser.parse_args(["teleop", "--wrist-roll-offset-deg", "-45.0"])
    assert args.wrist_roll_offset_deg == -45.0


def test_run_teleop_invokes_pre_dispatch_then_app_main(monkeypatch):
    """run_teleop calls pre_dispatch (e.g., MUJOCO_GL setup, eager imports),
    then forwards parsed args to teleop.app.main with the named backend."""
    called: dict = {}

    def _fake_pre_dispatch():
        called["pre_dispatch"] = True

    def _fake_app_main(args, backend: str):
        called["args"] = args
        called["backend"] = backend

    import so101_nexus_core.teleop.app as app_mod

    monkeypatch.setattr(app_mod, "main", _fake_app_main)
    monkeypatch.setattr("sys.argv", ["test-prog", "teleop"])

    run_teleop("mujoco", pre_dispatch=_fake_pre_dispatch)

    assert called["pre_dispatch"] is True
    assert called["backend"] == "mujoco"
    assert called["args"].command == "teleop"


def test_run_teleop_works_without_pre_dispatch(monkeypatch):
    """Guard: pre_dispatch=None must not crash; app.main is still invoked with the right backend."""
    called: dict = {}

    def _fake_app_main(args, backend: str):
        called["backend"] = backend

    import so101_nexus_core.teleop.app as app_mod

    monkeypatch.setattr(app_mod, "main", _fake_app_main)
    monkeypatch.setattr("sys.argv", ["test-prog", "teleop"])

    run_teleop("maniskill")

    assert called["backend"] == "maniskill"
