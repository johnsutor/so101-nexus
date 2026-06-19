"""Shared CLI test contracts for backend entry point modules."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from so101_nexus.env_ids import Backend


def run_parser_contract(
    cli_module: Any,
    *,
    expected_default_leader_id: str = "so101_leader",
) -> None:
    """Verify a backend parser accepts the shared teleop flags."""
    parser = cli_module._build_parser()

    args = parser.parse_args(["teleop", "--leader-port", "/dev/null"])
    assert args.command == "teleop"
    assert args.leader_port == "/dev/null"
    assert args.leader_id == expected_default_leader_id

    args = parser.parse_args(["teleop", "--wrist-roll-offset-deg", "-45.0"])
    assert args.wrist_roll_offset_deg == -45.0


def run_parser_requires_subcommand(cli_module: Any, *, pytest: Any) -> None:
    """Verify a backend parser rejects bare invocation."""
    parser = cli_module._build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([])


def run_main_dispatch_contract(
    cli_module: Any,
    *,
    backend: Backend,
    argv0: str,
    monkeypatch: Any,
) -> None:
    """Verify ``main()`` dispatches teleop to the shared app for *backend*."""
    called: dict[str, object] = {}

    def _fake_teleop_main(args: Any, **kwargs: Any) -> None:
        called["command"] = args.command
        called["backend"] = kwargs["backend"]

    import so101_nexus.teleop.app as app_mod

    monkeypatch.setattr(app_mod, "main", _fake_teleop_main)
    monkeypatch.setattr("sys.argv", [argv0, "teleop"])

    cli_module.main()

    assert called["backend"] == backend
    assert called["command"] == "teleop"
