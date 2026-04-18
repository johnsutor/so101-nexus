"""Tests for the maniskill-backend argparse CLI."""

from __future__ import annotations

import pytest

from so101_nexus_maniskill import cli as maniskill_cli


def test_build_parser_has_teleop_subcommand():
    parser = maniskill_cli._build_parser()
    args = parser.parse_args(["teleop", "--leader-port", "/dev/null"])
    assert args.command == "teleop"
    assert args.leader_port == "/dev/null"
    assert args.leader_id == "so101_leader"


def test_build_parser_requires_subcommand():
    parser = maniskill_cli._build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_build_parser_wrist_roll_offset_parses():
    parser = maniskill_cli._build_parser()
    args = parser.parse_args(["teleop", "--wrist-roll-offset-deg", "-45.0"])
    assert args.wrist_roll_offset_deg == -45.0


def test_main_dispatches_teleop(monkeypatch):
    called = {}

    def _fake_teleop_main(args, backend: str):
        called["args"] = args
        called["backend"] = backend

    import so101_nexus_core.teleop.app as app_mod

    monkeypatch.setattr(app_mod, "main", _fake_teleop_main)
    monkeypatch.setattr("sys.argv", ["so101-nexus-maniskill", "teleop"])

    maniskill_cli.main()

    assert called["backend"] == "maniskill"
    assert called["args"].command == "teleop"
