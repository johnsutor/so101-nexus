"""Regression tests for teleop initialization resource cleanup."""

from unittest.mock import Mock

import pytest

from so101_nexus.teleop import app


@pytest.mark.parametrize("finalize_fails", [False, True])
def test_finalization_finishes_inference_before_reusing_or_discarding_policy(
    monkeypatch, finalize_fails
):
    gr = pytest.importorskip("gradio")
    intervention = Mock(leader_released=True)
    policy = object()
    session = {
        "intervention": intervention,
        "collection_policy": policy,
        "leader": Mock(),
        "dataset": Mock(),
    }
    monkeypatch.setattr(app, "_write_env_config_meta", lambda _: None)
    if finalize_fails:
        session["dataset"].finalize.side_effect = RuntimeError("dataset failed")
        with pytest.raises(gr.Error, match="dataset failed"):
            app._cb_finalize_and_close(session)
        assert session["collection_policy"] is policy
    else:
        app._cb_finalize_and_close(session)
        assert "collection_policy" not in session
    intervention.close.assert_called_once_with(wait=True)
    assert "intervention" not in session


@pytest.mark.parametrize("first_failure", [False, True])
def test_reinitialization_reuses_policy_after_reconfiguration_or_failure(
    monkeypatch, first_failure
):
    from dataclasses import replace
    from types import SimpleNamespace

    policy = SimpleNamespace(reset=lambda: None, select_action=lambda batch: None, fps=30)
    load = Mock(return_value=policy)
    dataset = Mock(
        side_effect=[RuntimeError("dataset failed") if first_failure else Mock(), Mock()]
    )
    monkeypatch.setattr(app, "import_backend_for_env_id", lambda _: None)
    leader = Mock()
    monkeypatch.setattr(app, "_connect_leader", lambda *_: leader)
    monkeypatch.setattr(app, "load_collection_policy", load)
    monkeypatch.setattr(app, "_resolve_env_state_names", lambda *_, **__: [])
    monkeypatch.setattr(app, "_create_dataset", dataset)
    config = app.RecordingOptions(
        env_id="MuJoCoTouch-v1",
        robot_type="so101",
        leader_id="leader",
        fps=30,
        wrist_wh=(4, 4),
        overhead_wh=(4, 4),
        repo_id="local/test",
        num_episodes=1,
        action_space="joint_pos",
        max_steps=10,
        countdown=0,
        wrist_roll_offset_deg=-90,
        field_selection=app.FieldSelection(),
        env_overrides=None,
    )
    session = {"policy_options": {"factory": "example:build"}}
    try:
        first = {}
        app._run_init_worker(session, first, "/dev/null", config)
        assert first.get("error") == ("dataset failed" if first_failure else None)
        old = session.get("intervention")
        app._close_recording_worker(session, wait=True)
        second = {}
        app._run_init_worker(session, second, "/dev/null", replace(config, wrist_wh=(8, 8)))
        assert second["done"]
        assert not second.get("error")
        assert session["intervention"] is not old
        assert session["intervention"]._worker.policy is policy
        leader.get_action.assert_not_called()
        assert session["wrist_wh"] == (8, 8)
        load.assert_called_once_with(factory="example:build")

        app._close_recording_worker(session, wait=True)
        mismatch = {}
        app._run_init_worker(session, mismatch, "/dev/null", replace(config, fps=15))
        assert mismatch["error"]
        assert dataset.call_count == 2
        assert load.call_count == 1
    finally:
        app._close_recording_worker(session, wait=True)


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
        app.RecordingOptions(
            env_id="MuJoCoTouch-v1",
            robot_type="so101",
            leader_id="leader",
            fps=30,
            wrist_wh=(320, 240),
            overhead_wh=(640, 360),
            repo_id="local/test",
            num_episodes=1,
            action_space="joint_pos",
            max_steps=10,
            countdown=0,
            wrist_roll_offset_deg=-90.0,
            field_selection=app.FieldSelection(),
            env_overrides=None,
        ),
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
        app.RecordingOptions(
            env_id="MuJoCoTouch-v1",
            robot_type="so101",
            leader_id="leader",
            fps=30,
            wrist_wh=(4, 4),
            overhead_wh=(4, 4),
            repo_id="local/correction",
            num_episodes=1,
            action_space="joint_pos",
            max_steps=10,
            countdown=0,
            wrist_roll_offset_deg=-90,
            field_selection=app.FieldSelection(),
            env_overrides=None,
        ),
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


def test_duplicate_initialization_is_rejected_before_resource_changes(monkeypatch):
    gr = pytest.importorskip("gradio")
    close = Mock()
    launch = Mock()
    monkeypatch.setattr(app, "_close_recording_worker", close)
    monkeypatch.setattr(app.threading, "Thread", launch)
    with pytest.raises(gr.Error, match="already running"):
        app._start_init_attempt({}, {"running": True}, "/dev/null", {})
    close.assert_not_called()
    launch.assert_not_called()


@pytest.mark.parametrize("failed_attempt", [False, True])
def test_initialization_starts_once_and_allows_retry_after_failure(monkeypatch, failed_attempt):
    gr = pytest.importorskip("gradio")
    close = Mock()
    thread = Mock()
    monkeypatch.setattr(app, "_close_recording_worker", close)
    monkeypatch.setattr(app.threading, "Thread", thread)
    monkeypatch.setattr(app, "check_robot_env_mismatch", lambda *_: None)
    config = app.RecordingOptions(
        env_id="MuJoCoTouch-v1",
        robot_type="so101",
        leader_id="leader",
        fps=30,
        wrist_wh=(320, 240),
        overhead_wh=(640, 360),
        repo_id="local/test",
        num_episodes=1,
        action_space="joint_pos",
        max_steps=10,
        countdown=0,
        wrist_roll_offset_deg=-90,
        field_selection=app.FieldSelection(),
        env_overrides=None,
    )
    session = {}
    state = {
        "running": False,
        "done": failed_attempt,
        "error": "failure" if failed_attempt else None,
    }
    app._start_init_attempt(session, state, "/dev/null", config)
    assert thread.call_args.kwargs["args"] == (session, state, "/dev/null", config)
    assert state["last_config"] is config
    assert state["running"] is True
    assert state["done"] is False
    assert state["error"] is None
    with pytest.raises(gr.Error, match="already running"):
        app._start_init_attempt(session, state, "/dev/null", config)
    close.assert_called_once_with(session, wait=True)
    thread.return_value.start.assert_called_once_with()
