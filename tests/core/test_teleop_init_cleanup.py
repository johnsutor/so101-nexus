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


def test_policy_initialization_failure_closes_serial_without_disabling_torque(monkeypatch):
    from types import SimpleNamespace

    leader = Mock()
    policy = SimpleNamespace(reset=lambda: None, select_action=lambda batch: None, fps=30)
    sessions = []
    original = app.InterventionSession

    def create(*args, **kwargs):
        session = original(*args, **kwargs)
        sessions.append(session)
        return session

    monkeypatch.setattr(app, "import_backend_for_env_id", lambda _: None)
    monkeypatch.setattr(app, "_connect_leader", lambda *_: leader)
    monkeypatch.setattr(app, "load_collection_policy", lambda **kwargs: policy)
    monkeypatch.setattr(app, "InterventionSession", create)
    monkeypatch.setattr(app, "_resolve_env_state_names", lambda *_, **__: [])
    features = []

    def create_dataset(repo_id, fps, robot_type, schema):
        features.append(schema)
        raise RuntimeError("dataset creation failed")

    monkeypatch.setattr(app, "_create_dataset", create_dataset)
    session = {
        "policy_options": {"factory": "example:build"},
        "leader_alignment_options": {"joint_tolerance_deg": 12, "gripper_tolerance_percent": 6},
    }
    init = {}
    app._run_init_worker(
        session,
        init,
        "/dev/null",
        "MuJoCoTouch-v1",
        "so101",
        "leader",
        30,
        (4, 4),
        (4, 4),
        "local/correction",
        1,
        "joint_pos",
        10,
        0,
        -90,
        app.FieldSelection(),
        None,
    )
    assert init["error"] == "dataset creation failed"
    assert {"is_intervention", "intervention_id"} <= set(features[0])
    leader.disconnect.assert_not_called()
    leader.bus.disconnect.assert_called_once_with(disable_torque=False)
    assert len(sessions) == 1
    assert sessions[0].synchronizer.joint_tolerance_deg == 12
    assert sessions[0].synchronizer.gripper_tolerance_percent == 6
    with pytest.raises(RuntimeError, match="closed"):
        sessions[0].request("human")
