"""Tests for persistent simulator ownership during recording sessions."""

from __future__ import annotations

import threading
import types
from unittest.mock import Mock

import pytest

from so101_nexus.config import SO101_JOINT_NAMES
from so101_nexus.teleop.recorder import RecordingState, RecordingWorker
from tests.core.test_teleop_recorder import _FakeLeader, _FakeRecordingFollower


@pytest.fixture
def worker_setup(monkeypatch):
    import so101_nexus.lerobot_adapter.sim_follower as follower_module
    import so101_nexus.teleop.session as session_module

    events = []

    class Follower(_FakeRecordingFollower):
        instances = []

        def connect(self):
            events.append(("connect", threading.get_ident(), self.config.seed))

        def reset(self, *, seed):
            events.append(("reset", threading.get_ident(), seed))
            self.step_calls = 0
            self._last_step_info = None
            self._env.unwrapped.task_description = f"Task {seed}"

        def disconnect(self):
            events.append(("disconnect", threading.get_ident(), self.config.seed))

    monkeypatch.setattr(follower_module, "SimSOFollower", Follower)
    monkeypatch.setattr(session_module, "prepare_follower_calibration", lambda **kwargs: None)
    monkeypatch.setattr(session_module, "build_sim_follower_config", types.SimpleNamespace)
    kwargs = {
        "state": RecordingState(num_episodes=3),
        "env_id": "FakeEnv-v0",
        "leader": _FakeLeader(),
        "joint_names": SO101_JOINT_NAMES,
        "fps": 1000,
        "max_steps": 1,
        "countdown": 0,
        "wrist_roll_offset_deg": 0.0,
        "wrist_wh": (4, 4),
        "overhead_wh": (4, 4),
        "follower_calibration_dir": "unused",
        "follower_robot_id": "worker_test",
        "seed": 10,
    }
    return kwargs, Follower, events


def test_worker_reuses_follower_and_resets_episode_state(worker_setup):
    kwargs, follower_type, events = worker_setup
    state = kwargs["state"]
    worker = RecordingWorker()
    try:
        worker.submit(**kwargs)
        assert worker.wait(5)
        assert state.error is None
        assert state.episode_seed == 10
        first = follower_type.instances[0]
        state.episodes_completed = 1
        kwargs["leader"].get_action = lambda: {f"{name}.pos": 25.0 for name in SO101_JOINT_NAMES}
        worker.submit(**kwargs)
        assert worker.wait(5)
        assert state.error is None
        assert state.episode_seed == 11
        assert state.task_description == "Task 11"
        assert len(state.episode_actions) == 1
        assert state.episode_states[0][0] == 0.0
        assert first.initial_leader_action["shoulder_pan.pos"] == 25.0
        worker.submit(**kwargs)
        assert worker.wait(5)
        assert state.episode_seed == 11
        assert len(follower_type.instances) == 1
    finally:
        worker.close()
    assert [event[0] for event in events] == ["connect", "reset", "reset", "disconnect"]
    assert len({event[1] for event in events}) == 1
    assert events[0][1] != threading.get_ident()


@pytest.mark.parametrize("customization", ["env_config_factory", "env_config_profile", "env_id"])
def test_worker_rebuilds_for_dynamic_or_changed_configuration(worker_setup, customization):
    kwargs, follower_type, events = worker_setup
    worker = RecordingWorker()
    try:
        worker.submit(**kwargs)
        assert worker.wait(5)
        kwargs[customization] = lambda *args: (
            None if customization == "env_config_factory" else "changed"
        )
        worker.submit(**kwargs)
        assert worker.wait(5)
        worker.submit(**kwargs)
        assert worker.wait(5)
        assert kwargs["state"].error is None
        expected = 2 if customization == "env_id" else 3
        assert len(follower_type.instances) == expected
    finally:
        worker.close()
    assert len({event[1] for event in events}) == 1


def test_worker_failure_disconnects_and_retry_rebuilds(worker_setup):
    kwargs, follower_type, events = worker_setup
    original = kwargs["leader"].get_action
    worker = RecordingWorker()
    try:
        kwargs["leader"].get_action = lambda: (_ for _ in ()).throw(RuntimeError("broken"))
        worker.submit(**kwargs)
        assert worker.wait(5)
        assert "broken" in kwargs["state"].error
        assert events[-1][0] == "disconnect"
        kwargs["leader"].get_action = original
        worker.submit(**kwargs)
        assert worker.wait(5)
        assert kwargs["state"].error is None
        assert len(follower_type.instances) == 2
    finally:
        worker.close()


def test_worker_close_cancels_countdown_and_rejects_new_jobs(worker_setup):
    kwargs, _, events = worker_setup
    kwargs["countdown"] = 60
    worker = RecordingWorker()
    worker.submit(**kwargs)
    worker.close()
    assert kwargs["state"].recording_finished
    assert events == []
    with pytest.raises(RuntimeError, match="closed"):
        worker.submit(**kwargs)
    worker.close()


@pytest.mark.parametrize("acquisition", ["connect", "reset"])
def test_worker_stop_during_acquisition_does_not_begin_control(
    worker_setup, monkeypatch, acquisition
):
    kwargs, follower_type, _ = worker_setup
    intervention = Mock(intervention_id=0)
    intervention.action.side_effect = lambda obs, task, human: human()
    intervention.execute.side_effect = lambda action, send: send(action)
    kwargs["intervention"] = intervention
    entered, release = threading.Event(), threading.Event()
    worker = RecordingWorker()
    try:
        if acquisition == "reset":
            worker.submit(**kwargs)
            assert worker.wait(5)
            intervention.reset_mock()

        original = getattr(follower_type, acquisition)

        def blocked_acquire(self, **options):
            entered.set()
            assert release.wait(5)
            original(self, **options)

        monkeypatch.setattr(follower_type, acquisition, blocked_acquire)
        worker.submit(**kwargs)
        assert entered.wait(5)
        kwargs["state"].should_stop = True
        release.set()
        assert worker.wait(5)

        intervention.begin_episode.assert_not_called()
        intervention.action.assert_not_called()
        intervention.execute.assert_not_called()
        intervention.end_episode.assert_called_once()
        assert follower_type.instances[-1].step_calls == 0
        assert kwargs["state"].recording_finished
        assert kwargs["state"].error is None
    finally:
        release.set()
        worker.close()


def test_worker_rejects_overlapping_recordings_and_stops_active_episode(worker_setup):
    kwargs, _, events = worker_setup
    entered = threading.Event()
    release = threading.Event()
    leader_action = kwargs["leader"].get_action
    reads = 0

    def get_action():
        nonlocal reads
        reads += 1
        if reads == 2:
            entered.set()
            assert release.wait(5)
        return leader_action()

    kwargs["leader"].get_action = get_action
    kwargs["max_steps"] = 100
    worker = RecordingWorker()
    worker.submit(**kwargs)
    assert entered.wait(5)
    try:
        with pytest.raises(RuntimeError, match="already recording"):
            worker.submit(**kwargs)
        kwargs["state"].should_stop = True
        release.set()
        assert worker.wait(5)
        assert len(kwargs["state"].episode_actions) == 1
    finally:
        release.set()
        worker.close()
    assert [event[0] for event in events] == ["connect", "disconnect"]
