"""Tests for shared teleop CLI helpers."""

from __future__ import annotations

import pytest

from so101_nexus.teleop.cli import TeleopArgs, parse_teleop_args, run_teleop


def test_parse_teleop_args_accepts_shared_flags() -> None:
    args = parse_teleop_args(["teleop", "--leader-port", "/dev/null", "--leader-id", "leader"])

    assert isinstance(args, TeleopArgs)
    assert args.leader_port == "/dev/null"
    assert args.leader_id == "leader"


def test_parse_teleop_args_requires_subcommand() -> None:
    with pytest.raises(SystemExit):
        parse_teleop_args([])


def test_parse_teleop_args_wrist_roll_offset_parses() -> None:
    args = parse_teleop_args(["teleop", "--wrist-roll-offset-deg", "-45.0"])

    assert args.wrist_roll_offset_deg == -45.0


def test_parse_teleop_args_accepts_env_customization_flags() -> None:
    args = parse_teleop_args(
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
        calls.append(f"main:{backend}:{type(args).__name__}")

    import so101_nexus.teleop.app as app_mod

    monkeypatch.setattr(app_mod, "main", _fake_main)
    monkeypatch.setattr("sys.argv", ["so101-nexus-test", "teleop"])

    run_teleop("mujoco", pre_dispatch=_setup)

    assert calls == ["setup", "main:mujoco:TeleopArgs"]


def test_run_teleop_works_without_setup(monkeypatch) -> None:
    called: dict[str, object] = {}

    def _fake_main(args, backend: str) -> None:
        called["args_type"] = type(args).__name__
        called["backend"] = backend

    import so101_nexus.teleop.app as app_mod

    monkeypatch.setattr(app_mod, "main", _fake_main)
    monkeypatch.setattr("sys.argv", ["so101-nexus-test", "teleop"])

    run_teleop("mujoco")

    assert called == {"args_type": "TeleopArgs", "backend": "mujoco"}
