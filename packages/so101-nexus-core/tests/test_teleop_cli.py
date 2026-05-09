"""Tests for shared teleop CLI helpers."""

from __future__ import annotations

import pytest

from so101_nexus_core.teleop.cli import build_teleop_parser, run_teleop


def test_build_teleop_parser_accepts_shared_flags() -> None:
    parser = build_teleop_parser(prog="so101-nexus-test")

    args = parser.parse_args(["teleop", "--leader-port", "/dev/null", "--leader-id", "leader"])

    assert args.command == "teleop"
    assert args.leader_port == "/dev/null"
    assert args.leader_id == "leader"


def test_build_teleop_parser_requires_subcommand() -> None:
    parser = build_teleop_parser(prog="so101-nexus-test")

    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_build_teleop_parser_wrist_roll_offset_parses() -> None:
    parser = build_teleop_parser(prog="so101-nexus-test")

    args = parser.parse_args(["teleop", "--wrist-roll-offset-deg", "-45.0"])

    assert args.wrist_roll_offset_deg == -45.0


def test_build_teleop_parser_accepts_env_customization_flags() -> None:
    parser = build_teleop_parser(prog="so101-nexus-test")

    args = parser.parse_args(
        [
            "teleop",
            "--env-config-profile",
            "profile.toml",
            "--env-config-factory",
            "my_mod:build",
            "--env-module",
            "my_custom_envs",
            "--extra-env-id",
            "CustomPick-v1",
        ]
    )

    assert args.env_config_profile == "profile.toml"
    assert args.env_config_factory == "my_mod:build"
    assert args.env_modules == ["my_custom_envs"]
    assert args.extra_env_ids == ["CustomPick-v1"]


def test_run_teleop_invokes_setup_before_app_main(monkeypatch) -> None:
    calls: list[str] = []

    def _setup() -> None:
        calls.append("setup")

    def _fake_main(args, backend: str) -> None:
        calls.append(f"main:{backend}:{args.command}")

    import so101_nexus_core.teleop.app as app_mod

    monkeypatch.setattr(app_mod, "main", _fake_main)
    monkeypatch.setattr("sys.argv", ["so101-nexus-test", "teleop"])

    run_teleop("mujoco", pre_dispatch=_setup)

    assert calls == ["setup", "main:mujoco:teleop"]


def test_run_teleop_works_without_setup(monkeypatch) -> None:
    called: dict[str, object] = {}

    def _fake_main(args, backend: str) -> None:
        called["command"] = args.command
        called["backend"] = backend

    import so101_nexus_core.teleop.app as app_mod

    monkeypatch.setattr(app_mod, "main", _fake_main)
    monkeypatch.setattr("sys.argv", ["so101-nexus-test", "teleop"])

    run_teleop("maniskill")

    assert called == {"command": "teleop", "backend": "maniskill"}
