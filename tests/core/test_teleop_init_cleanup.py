"""Regression tests for teleop initialization resource cleanup."""

from unittest.mock import Mock

import pytest

from so101_nexus.teleop import app


@pytest.mark.parametrize(
    "failure_stage", ["_resolve_env_state_names", "build_features", "_create_dataset"]
)
@pytest.mark.parametrize("disconnect_fails", [False, True])
def test_init_failure_disconnects_connected_leader(
    monkeypatch, failure_stage, disconnect_fails
) -> None:
    leader = Mock()
    if disconnect_fails:
        leader.disconnect.side_effect = RuntimeError("serial port already closed")
    monkeypatch.setattr(app, "import_backend_for_env_id", lambda _: None)
    monkeypatch.setattr(app, "_connect_leader", lambda *_: leader)
    monkeypatch.setattr(app, "_resolve_env_state_names", lambda *_, **__: [])

    def fail(*_, **__):
        raise ValueError("invalid recording config")

    monkeypatch.setattr(app, failure_stage, fail)
    session: dict = {}
    init_state: dict = {}
    app._run_init_worker(
        session,
        init_state,
        "/dev/ttyACM0",
        "MuJoCoTouch-v1",
        "so101",
        "leader",
        30,
        (320, 240),
        (640, 360),
        "local/test",
        1,
        "joint_pos",
        10,
        0,
        -90.0,
        app.FieldSelection(),
        None,
    )

    assert init_state["done"] is True
    assert init_state["error"] == "invalid recording config"
    assert "leader" not in session
    leader.disconnect.assert_called_once_with()
