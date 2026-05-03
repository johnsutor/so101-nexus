"""Shared CLI test contract for backend `cli.py` modules.

Each backend's `test_cli.py` reduces to a few `run_*` calls plus any
backend-specific assertions (e.g., the MuJoCo `MUJOCO_GL` env-var test).
"""

from __future__ import annotations

from typing import Any


def run_parser_contract(
    cli_module: Any, *, expected_default_leader_id: str = "so101_leader"
) -> None:
    """Verify the parser accepts the standard `teleop --leader-port` invocation."""
    parser = cli_module._build_parser()
    args = parser.parse_args(["teleop", "--leader-port", "/dev/null"])
    assert args.command == "teleop"
    assert args.leader_port == "/dev/null"
    assert args.leader_id == expected_default_leader_id


def run_parser_wrist_roll_offset_contract(cli_module: Any) -> None:
    """Verify the parser accepts the `--wrist-roll-offset-deg` flag."""
    parser = cli_module._build_parser()
    args = parser.parse_args(["teleop", "--wrist-roll-offset-deg", "-45.0"])
    assert args.wrist_roll_offset_deg == -45.0


def run_parser_requires_subcommand(cli_module: Any) -> None:
    """Bare invocation must raise SystemExit (argparse `required=True`)."""
    import pytest

    parser = cli_module._build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def run_main_dispatches_teleop(
    cli_module: Any,
    *,
    backend: str,
    monkeypatch: Any,
    argv_prog: str,
) -> dict:
    """Invoke `cli_module.main()` with a stubbed `teleop.app.main` and return what was captured."""
    called: dict = {}

    def _fake_app_main(args, backend: str):
        called["args"] = args
        called["backend"] = backend

    import so101_nexus_core.teleop.app as app_mod

    monkeypatch.setattr(app_mod, "main", _fake_app_main)
    monkeypatch.setattr("sys.argv", [argv_prog, "teleop"])

    cli_module.main()

    assert called["backend"] == backend
    assert called["args"].command == "teleop"
    return called
