"""Tests for policy collection with interruptible human control."""

from __future__ import annotations

import threading
import time
from itertools import pairwise
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from so101_nexus import SO101_JOINT_NAMES
from so101_nexus.teleop.intervention import InterventionSession, intervention_chunk_mask
from so101_nexus.teleop.leader import SOLeaderSynchronizer
from tests.core.test_teleop_worker import worker_setup  # noqa: F401


def observation(value=0.0):
    return {f"{name}.pos": value for name in SO101_JOINT_NAMES}


class Policy:
    def __init__(self):
        self.resets = 0
        self.batches = []

    def reset(self):
        self.resets += 1

    def select_action(self, batch):
        self.batches.append(batch)
        return np.full(6, 10.0)


class Synchronizer:
    def __init__(self):
        self.targets = []
        self.cancels = 0
        self.releases = 0
        self.ready = False
        self.holding = True
        self.parking = False

    def start(self, target, *, duration_s=None):
        self.targets.append(target)
        self.parking = duration_s is not None

    def tick(self):
        return self.parking or self.ready

    def cancel(self):
        self.cancels += 1
        self.holding = True
        self.parking = False

    def release(self):
        self.releases += 1
        self.holding = False


def await_action(session, obs):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        action = session.action(obs, "Pick", lambda: observation(3.0))
        if action is not None:
            return action
        time.sleep(0.001)
    pytest.fail("policy did not produce an action")


def test_policy_human_policy_handoff_uses_fresh_observation():
    policy, sync = Policy(), Synchronizer()
    session = InterventionSession(policy, sync)
    try:
        session.begin_episode()
        obs = {**observation(), "wrist": np.zeros((2, 2, 3), dtype=np.uint8)}
        assert await_action(session, obs)["shoulder_pan.pos"] == 10
        session.request("human")
        assert session.action(obs, "Pick", lambda: observation(3)) is None
        assert session.status == "synchronizing"
        assert sync.targets == [observation()]
        sync.ready = True
        action = session.action(obs, "Pick", lambda: observation(3))
        assert action == observation(3)
        assert session.source == "human"
        assert session.intervention_id == 1
        session.request("policy")
        assert await_action(session, observation(7))["shoulder_pan.pos"] == 10
        assert policy.resets == 2
        np.testing.assert_array_equal(policy.batches[-1]["observation.state"], np.full(6, 7))
        assert policy.batches[0]["observation.images.wrist"].shape == (2, 2, 3)
        assert session.source == "policy"
    finally:
        session.close()


def test_takeover_and_close_do_not_wait_for_blocked_inference():
    entered, release = threading.Event(), threading.Event()

    class BlockingPolicy(Policy):
        def select_action(self, batch):
            entered.set()
            assert release.wait(5)
            return super().select_action(batch)

    sync = Synchronizer()
    sync.ready = True
    session = InterventionSession(BlockingPolicy(), sync)
    try:
        session.begin_episode()
        assert session.action(observation(), "Pick", lambda: observation(3)) is None
        assert entered.wait(2)
        session.request("human")
        assert session.action(observation(), "Pick", lambda: observation(3)) == observation(3)
        session.close()
        assert sync.cancels >= 1
    finally:
        release.set()
        session.close()


def test_close_can_wait_for_inference_after_nonblocking_close():
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()

    class BlockingPolicy(Policy):
        def select_action(self, batch):
            entered.set()
            assert release.wait(5)
            return super().select_action(batch)

    session = InterventionSession(BlockingPolicy(), Synchronizer())

    def wait_for_close():
        try:
            session.close(wait=True)
        finally:
            closed.set()

    waiter = threading.Thread(target=wait_for_close)
    try:
        session.begin_episode()
        session.action(observation(), "Pick", lambda: observation(3))
        assert entered.wait(2)
        session.close()
        waiter.start()
        assert not closed.wait(0.05)
        release.set()
        assert closed.wait(2)
        waiter.join(2)
        assert not session._worker.thread.is_alive()
    finally:
        release.set()
        if waiter.ident is not None:
            waiter.join(2)
        session.close()


def test_hold_cancels_sync_and_repeated_takeovers_get_distinct_ids():
    sync = Synchronizer()
    session = InterventionSession(Policy(), sync)
    try:
        session.begin_episode()
        session.request("human")
        session.action(observation(), "Pick", lambda: observation(3))
        session.request("hold")
        assert session.action(observation(), "Pick", lambda: observation(3)) is None
        assert session.status == "hold"
        assert sync.cancels >= 1
        sync.ready = True
        for expected in (1, 2):
            session.request("human")
            assert session.action(observation(), "Pick", lambda: observation(3)) is not None
            assert session.intervention_id == expected
            session.request("hold")
            session.action(observation(), "Pick", lambda: observation(3))
    finally:
        session.close()


@pytest.mark.parametrize("output", [np.zeros(5), np.full(6, np.nan), np.full(6, np.inf)])
def test_invalid_policy_actions_are_rejected(output):
    policy = Policy()
    policy.select_action = lambda batch: output
    session = InterventionSession(policy, Synchronizer())
    try:
        session.begin_episode()
        with pytest.raises(ValueError, match=r"finite.*6"):
            await_action(session, observation())
    finally:
        session.close()


@pytest.mark.parametrize("tensor", [False, True])
def test_chunk_mask_excludes_autonomy_and_crossing_intervention_boundaries(tensor):
    ids = np.array([0, 1, 1, 0, 2, 2, 2])
    if tensor:
        import torch

        ids = torch.as_tensor(ids)
    mask = intervention_chunk_mask(ids, 3)
    expected = [
        [False] * 3,
        [True, True, False],
        [True, False, False],
        [False] * 3,
        [True] * 3,
        [True, True, False],
        [True, False, False],
    ]
    if tensor:
        assert mask.device == ids.device
        mask = mask.numpy()
    np.testing.assert_array_equal(mask, expected)


class Bus:
    def __init__(self):
        self.positions = dict.fromkeys(SO101_JOINT_NAMES, 0.0)
        self.events = []

    def sync_read(self, register, *, num_retry=0):
        return dict(self.positions)

    def sync_write(self, register, values, *, num_retry=0):
        self.events.append(("write", dict(values)))
        self.positions.update(values)

    def write(self, register, motor, value, *, num_retry=0):
        self.events.append(("hold", motor, value))
        self.positions[motor] = value

    def enable_torque(self, *, motors=None, num_retry=0):
        self.events.append(("enable", motors))

    def disable_torque(self, *, num_retry=0):
        self.events.append(("disable",))


def test_timed_return_takes_four_seconds_with_gentle_endpoints(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    bus = Bus()
    sync = SOLeaderSynchronizer(bus, wrist_roll_offset_deg=0)
    sync.start(observation(80), duration_s=4.0)
    assert not sync.tick()
    for elapsed, expected in [(0.1, 0.145), (2.0, 40.0), (3.9, 79.855)]:
        clock[0] = elapsed
        assert not sync.tick()
        assert bus.positions["shoulder_pan"] == pytest.approx(expected, abs=0.01)
    clock[0] = 4.0
    assert sync.tick()
    assert bus.positions == dict.fromkeys(SO101_JOINT_NAMES, 80)
    assert sync.holding


@pytest.mark.parametrize("duration_s", [0, -1, float("nan"), float("inf")])
def test_timed_return_rejects_invalid_duration_before_motor_commands(duration_s):
    bus = Bus()
    sync = SOLeaderSynchronizer(bus, wrist_roll_offset_deg=0)
    with pytest.raises(ValueError, match="duration must be positive and finite"):
        sync.start(observation(), duration_s=duration_s)
    assert not bus.events


@pytest.mark.parametrize("operation,value", [("cancel", 1), ("release", 0)])
@pytest.mark.parametrize("persistent", [False, True])
def test_feetech_lock_acknowledgment_retries_are_bounded(monkeypatch, operation, value, persistent):
    pytest.importorskip("scservo_sdk")
    from lerobot.motors.feetech import FeetechMotorsBus

    from so101_nexus.lerobot_adapter.normalization import build_so101_motors

    bus = FeetechMotorsBus(port="/dev/null", motors=build_so101_motors(use_degrees=True))
    monkeypatch.setattr(bus.port_handler, "is_open", True)
    monkeypatch.setattr(
        bus, "sync_read", lambda *args, **kwargs: dict.fromkeys(SO101_JOINT_NAMES, 0.0)
    )
    monkeypatch.setattr(bus, "sync_write", lambda *args, **kwargs: None)
    monkeypatch.setattr(bus, "_unnormalize", lambda values: dict.fromkeys(values, 0))
    lock_address = bus.model_ctrl_table["sts3215"]["Lock"][0]
    torque_address = bus.model_ctrl_table["sts3215"]["Torque_Enable"][0]
    writes = []
    attempts = 0

    def write_packet(_port, motor_id, address, _length, data):
        nonlocal attempts
        writes.append((motor_id, address, data[0]))
        if motor_id == 1 and address == lock_address:
            attempts += 1
            if persistent or attempts == 1:
                return -6, 0
        return 0, 0

    monkeypatch.setattr(bus.packet_handler, "writeTxRx", write_packet)
    sync = SOLeaderSynchronizer(bus, wrist_roll_offset_deg=0)
    if persistent:
        with pytest.raises(ConnectionError, match=r"Lock.*after 4 tries"):
            getattr(sync, operation)()
        assert attempts == 4
        assert not sync.holding
        if operation == "cancel":
            assert (6, torque_address, 1) in writes
            assert (6, lock_address, 1) in writes
    else:
        getattr(sync, operation)()
        assert attempts == 2
        assert sync.holding is (operation == "cancel")
        for motor_id in range(1, 7):
            assert (motor_id, torque_address, value) in writes
            assert (motor_id, lock_address, value) in writes
    assert all(
        written == value
        for _, address, written in writes
        if address in (lock_address, torque_address)
    )


def test_leader_sync_interpolates_converts_offset_and_holds_until_explicit_release(monkeypatch):
    now = [0.0]
    monkeypatch.setattr("so101_nexus.teleop.leader.time.monotonic", lambda: now[0])
    bus = Bus()
    sync = SOLeaderSynchronizer(bus, wrist_roll_offset_deg=-90)
    target = observation(10)
    sync.start(target)
    assert bus.events[:12] == [
        event for name in SO101_JOINT_NAMES for event in (("hold", name, 0.0), ("enable", name))
    ]
    assert not sync.tick()
    now[0] = 1.0
    assert not sync.tick()
    assert bus.positions["wrist_roll"] == 20
    now[0] = 5.0
    assert sync.tick()
    assert bus.positions["wrist_roll"] == 100
    assert bus.positions["gripper"] == 10
    assert ("disable",) not in bus.events
    sync.release()
    assert bus.events[-1] == ("disable",)


def test_leader_sync_timeout_and_cancel_hold_measured_pose(monkeypatch):
    now = [0.0]
    monkeypatch.setattr("so101_nexus.teleop.leader.time.monotonic", lambda: now[0])
    bus = Bus()
    sync = SOLeaderSynchronizer(bus, wrist_roll_offset_deg=0)
    sync.start(observation(10))
    bus.sync_write = lambda *args, **kwargs: None
    now[0] = 10
    with pytest.raises(TimeoutError, match="synchron"):
        sync.tick()
    assert ("disable",) not in bus.events
    assert bus.events[-1] == ("enable", "gripper")
    sync.cancel()
    assert bus.events[-1] == ("enable", "gripper")
    assert ("disable",) not in bus.events


def test_late_policy_result_is_discarded_after_human_correction():
    entered, release = threading.Event(), threading.Event()
    policy = Policy()
    original = policy.select_action

    def select(batch):
        if not policy.batches:
            entered.set()
            assert release.wait(5)
        return original(batch)

    policy.select_action = select
    sync = Synchronizer()
    sync.ready = True
    session = InterventionSession(policy, sync)
    try:
        session.begin_episode()
        session.action(observation(), "Pick", lambda: observation(3))
        assert entered.wait(2)
        session.request("human")
        assert session.action(observation(2), "Pick", lambda: observation(3)) == observation(3)
        session.request("policy")
        release.set()
        await_action(session, observation(9))
        assert policy.resets == 2
        np.testing.assert_array_equal(policy.batches[-1]["observation.state"], np.full(6, 9))
    finally:
        release.set()
        session.close()


def test_request_between_selection_and_execution_prevents_step():
    session = InterventionSession(Policy(), Synchronizer())
    try:
        session.begin_episode()
        action = await_action(session, observation())
        session.request("hold")
        sent = []
        assert session.execute(action, sent.append) is None
        assert sent == []
    finally:
        session.close()


def test_intervention_dataset_features_and_frames_match():
    from so101_nexus.teleop.dataset import FieldSelection, build_features, build_frame

    selection = FieldSelection(wrist_image=False, overhead_image=False, environment_state=False)
    features = {f"{name}.pos": float for name in SO101_JOINT_NAMES}
    schema = build_features(selection, features, features, intervention=True)
    frame = build_frame(
        selection,
        state=np.zeros(6),
        action=np.ones(6),
        task="Pick",
        wrist_image=None,
        overhead_image=None,
        intervention_id=2,
    )
    assert set(frame) - {"task"} == set(schema)
    np.testing.assert_array_equal(frame["is_intervention"], [1])
    np.testing.assert_array_equal(frame["intervention_id"], [2])
    frame = build_frame(
        selection,
        state=np.zeros(6),
        action=np.ones(6),
        task="Pick",
        wrist_image=None,
        overhead_image=None,
        intervention_id=0,
    )
    np.testing.assert_array_equal(frame["is_intervention"], [0])


def test_recording_worker_records_executed_policy_actions(worker_setup):  # noqa: F811
    from so101_nexus.teleop.recorder import RecordingWorker

    kwargs, _, _ = worker_setup
    session = InterventionSession(Policy(), Synchronizer())
    worker = RecordingWorker()
    try:
        worker.submit(**kwargs, intervention=session)
        assert worker.wait(3)
        state = kwargs["state"]
        assert state.error is None
        np.testing.assert_array_equal(state.transitions[0].action, np.full(6, 10))
        assert [t.intervention_id for t in state.transitions] == [0]
        assert session.status == "hold"
    finally:
        worker.close()
        session.close()


@pytest.mark.parametrize("outcome", ["ready", "stop", "timeout"])
@pytest.mark.parametrize("released", [False, True])
def test_worker_returns_leader_before_reset_and_policy(worker_setup, outcome, released):  # noqa: F811
    from so101_nexus.teleop.recorder import RecordingWorker

    kwargs, follower_type, events = worker_setup
    sync = Synchronizer()
    rest = observation(12)
    follower_type.get_rest_action = lambda self, current: dict(rest)
    ticks = []

    def tick():
        assert len(follower_type.instances) == 1
        assert follower_type.instances[0].step_calls == 0
        assert session.status == "returning to rest"
        assert not kwargs["state"].transitions
        ticks.append(1)
        if outcome == "timeout":
            raise TimeoutError("rest position unreachable")
        if outcome == "stop":
            kwargs["state"].should_stop = True
            return False
        return len(ticks) == 2

    sync.tick = tick
    policy = Policy()
    session = InterventionSession(policy, sync)
    if released:
        assert session.release_leader()
    prior_cancels = sync.cancels
    worker = RecordingWorker()
    try:
        worker.submit(**kwargs, intervention=session)
        assert worker.wait(3)
        state = kwargs["state"]
        assert sync.targets == [observation(12)]
        assert sync.releases == int(released)
        assert sync.cancels > prior_cancels
        assert not session.leader_released
        assert sync.holding
        assert session.status == "hold"
        if outcome == "ready":
            assert state.error is None
            assert len(state.transitions) == 1
            assert policy.batches
        else:
            assert all(event[0] != "reset" for event in events)
            assert not policy.batches
            assert not state.transitions
            assert (state.error is None) == (outcome == "stop")
            if outcome == "timeout":
                assert "rest position unreachable" in state.error
    finally:
        worker.close()
        session.close()


def test_rest_return_ramps_physical_pose_and_preserves_wrist_offset(monkeypatch):
    from so101_nexus.teleop.recorder import RecordingState, _return_leader_to_rest

    clock = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    bus = Bus()
    rest = observation(30)
    rest["wrist_roll.pos"] = -107
    from so101_nexus.teleop.leader import apply_wrist_roll_offset_deg

    follower = SimpleNamespace(
        get_rest_action=lambda current: apply_wrist_roll_offset_deg(rest, -90),
        set_initial_leader_action=Mock(),
        reset=Mock(),
    )
    leader = SimpleNamespace(
        get_action=lambda: {f"{key}.pos": value for key, value in bus.positions.items()}
    )
    sync = SOLeaderSynchronizer(bus, wrist_roll_offset_deg=-90)
    session = InterventionSession(Policy(), sync)
    try:
        for _ in range(2):
            bus.positions = dict.fromkeys(SO101_JOINT_NAMES, 0.0)
            bus.events.clear()
            started = clock[0]
            assert _return_leader_to_rest(RecordingState(), follower, leader, session, -90)
            assert clock[0] - started == pytest.approx(4.0, abs=0.02)
            assert bus.positions == pytest.approx(
                {name.removesuffix(".pos"): value for name, value in rest.items()}
            )
            writes = [event[1] for event in bus.events if event[0] == "write"]
            assert len(writes) > 2
            for previous, current in pairwise(writes):
                for name in SO101_JOINT_NAMES:
                    peak_speed = 1.5 * abs(rest[f"{name}.pos"]) / 4
                    assert abs(current[name] - previous[name]) <= peak_speed * 0.02 + 1e-6
            assert sync.holding
            assert ("disable",) not in bus.events
    finally:
        session.close()


@pytest.mark.parametrize("rest", [{}, observation(float("nan"))])
def test_rest_return_rejects_invalid_pose_before_moving(rest):
    from so101_nexus.teleop.recorder import RecordingState, _return_leader_to_rest

    sync = Synchronizer()
    session = InterventionSession(Policy(), sync)
    try:
        with pytest.raises(ValueError, match="complete, finite"):
            _return_leader_to_rest(
                RecordingState(), Mock(), SimpleNamespace(get_action=lambda: rest), session, 0
            )
        assert not sync.targets
    finally:
        session.close()


@pytest.mark.filterwarnings("error:Conversion of an array with ndim:DeprecationWarning")
def test_intervention_fields_reload_from_real_lerobot_dataset(tmp_path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from so101_nexus.teleop.dataset import (
        FieldSelection,
        _make_reward_scalar_dataset_cls,
        build_features,
        build_frame,
    )

    selection = FieldSelection(wrist_image=False, overhead_image=False, environment_state=False)
    joints = {f"{name}.pos": float for name in SO101_JOINT_NAMES}
    features = build_features(selection, joints, joints, intervention=True)
    root = tmp_path / "corrections"
    dataset = _make_reward_scalar_dataset_cls().create(
        repo_id="local/corrections",
        fps=30,
        features=features,
        root=root,
        use_videos=False,
    )
    for segment in (0, 1, 1, 0, 2):
        dataset.add_frame(
            build_frame(
                selection,
                state=np.zeros(6),
                action=np.ones(6),
                task="Pick",
                wrist_image=None,
                overhead_image=None,
                intervention_id=segment,
            )
        )
    dataset.save_episode()
    dataset.finalize()
    reloaded = LeRobotDataset("local/corrections", root=root)
    assert len(reloaded) == 5
    assert [int(reloaded[i]["intervention_id"].item()) for i in range(5)] == [0, 1, 1, 0, 2]
    assert [bool(reloaded[i]["is_intervention"].item()) for i in range(5)] == [
        False,
        True,
        True,
        False,
        True,
    ]


def test_delta_policy_targets_are_anchored_to_current_readback():
    session = InterventionSession(Policy(), Synchronizer(), action_space="joint_pos_delta")
    try:
        session.begin_episode()
        assert await_action(session, observation(7)) == observation(17)
    finally:
        session.close()


def test_new_episode_discards_previous_predictions_and_resets_segment_ids():
    sync = Synchronizer()
    sync.ready = True
    policy = Policy()
    session = InterventionSession(policy, sync)
    try:
        session.begin_episode()
        session.request("human")
        session.action(observation(), "Pick", lambda: observation(3))
        assert session.intervention_id == 1
        session.end_episode()
        session.begin_episode()
        assert await_action(session, observation(4)) == observation(10)
        assert session.intervention_id == 0
        session.request("human")
        session.action(observation(), "Pick", lambda: observation(3))
        assert session.intervention_id == 1
    finally:
        session.close()


def test_worker_handoffs_preserve_continuous_episode_and_camera_alignment(worker_setup):  # noqa: F811
    from so101_nexus.teleop.recorder import RecordingWorker

    kwargs, follower_type, _ = worker_setup
    policy, sync = Policy(), Synchronizer()
    sync.ready = True
    session = InterventionSession(policy, sync)
    original = follower_type.get_observation

    def get_observation(follower):
        if follower.step_calls == 1:
            session.request("human")
        elif follower.step_calls == 2:
            session.request("policy")
        return original(follower)

    follower_type.get_observation = get_observation
    kwargs["max_steps"] = 3
    worker = RecordingWorker()
    try:
        worker.submit(**kwargs, intervention=session)
        assert worker.wait(3)
        state = kwargs["state"]
        assert state.error is None
        assert [t.intervention_id for t in state.transitions] == [
            0,
            1,
            0,
        ]
        assert len([t.wrist_image for t in state.transitions]) == len(state.transitions) == 3
        np.testing.assert_array_equal(
            [t.state for t in state.transitions], [np.full(6, i) for i in range(3)]
        )
        assert sync.targets == [kwargs["leader"].get_action(), observation(1)]
        assert policy.resets == 2
        np.testing.assert_array_equal(policy.batches[-1]["observation.state"], np.full(6, 2))
    finally:
        worker.close()
        session.close()


@pytest.mark.parametrize("dtype_name", ["float32", "bfloat16"])
def test_torch_policy_outputs_convert_to_follower_targets(dtype_name):
    import torch

    policy = Policy()
    policy.select_action = lambda batch: torch.full((6,), 12.0, dtype=getattr(torch, dtype_name))
    session = InterventionSession(policy, Synchronizer())
    try:
        session.begin_episode()
        assert await_action(session, observation()) == observation(12)
    finally:
        session.close()


def test_paused_takeover_waits_for_explicit_correction_start():
    sync = Synchronizer()
    session = InterventionSession(Policy(), sync)
    try:
        session.begin_episode()
        session.request("human", pause_after_sync=True)
        assert session.action(observation(), "Pick", lambda: pytest.fail("human read")) is None
        assert session.status == "synchronizing"
        sync.ready = True
        for _ in range(3):
            assert session.action(observation(), "Pick", lambda: pytest.fail("human read")) is None
            assert session.status == "human paused"
            assert session.source == "hold"
        session.request("human")
        assert session.action(observation(), "Pick", lambda: observation(3)) == observation(3)
        assert session.intervention_id == 1
        assert sync.targets == [observation()]
    finally:
        session.close()


@pytest.mark.parametrize("destination", ["hold", "policy"])
def test_paused_takeover_can_be_canceled(destination):
    sync = Synchronizer()
    sync.ready = True
    session = InterventionSession(Policy(), sync)
    try:
        session.begin_episode()
        session.request("human", pause_after_sync=True)
        assert session.action(observation(), "Pick", lambda: pytest.fail("human read")) is None
        session.request(destination)
        if destination == "policy":
            assert await_action(session, observation()) == observation(10)
        else:
            assert session.action(observation(), "Pick", lambda: pytest.fail("human read")) is None
            assert session.status == "hold"
    finally:
        session.close()


def test_paused_takeover_does_not_record_or_step_simulation():
    from types import SimpleNamespace

    from so101_nexus.teleop.recorder import RecordingState, _record_step

    sync = Synchronizer()
    sync.ready = True
    session = InterventionSession(Policy(), sync)
    follower = SimpleNamespace(
        get_observation=observation,
        send_action=lambda _: pytest.fail("simulation stepped while paused"),
    )
    state = RecordingState(num_episodes=1)
    try:
        session.begin_episode()
        session.request("human", pause_after_sync=True)
        for _ in range(5):
            assert not _record_step(state, follower, None, session, SO101_JOINT_NAMES, 0)
        assert [t.action for t in state.transitions] == []
        assert [t.intervention_id for t in state.transitions] == []
    finally:
        session.close()


@pytest.mark.parametrize("destination", ["hold", "policy"])
def test_stale_start_correction_cannot_override_pending_handoff(destination):
    sync = Synchronizer()
    sync.ready = True
    session = InterventionSession(Policy(), sync)
    try:
        session.begin_episode()
        assert not session.start_correction()
        session.request("human", pause_after_sync=True)
        assert not session.start_correction()
        session.action(observation(), "Pick", lambda: pytest.fail("human read"))
        session.request(destination)
        assert not session.start_correction()
        if destination == "policy":
            assert await_action(session, observation()) == observation(10)
        else:
            assert session.action(observation(), "Pick", lambda: pytest.fail("human read")) is None
    finally:
        session.close()


def test_start_correction_anchors_paused_leader_before_resuming():
    sync = Synchronizer()
    sync.ready = True
    session = InterventionSession(Policy(), sync)
    try:
        session.begin_episode()
        session.request("human", pause_after_sync=True)
        session.action(observation(), "Pick", lambda: pytest.fail("human read"))
        assert session.start_correction()
        assert not session.start_correction()
        assert session.action(observation(), "Pick", lambda: observation(3)) == observation()
        assert session.action(observation(), "Pick", lambda: observation(4)) == observation(1)
        assert sync.targets == [observation()]
        assert session.intervention_id == 1
    finally:
        session.close()


@pytest.mark.parametrize("destination", ["hold", "policy"])
def test_fresh_takeover_after_pending_handoff_stays_paused(destination):
    sync = Synchronizer()
    sync.ready = True
    session = InterventionSession(Policy(), sync)
    try:
        session.begin_episode()
        session.request("human", pause_after_sync=True)
        session.action(observation(), "Pick", lambda: pytest.fail("human read"))
        session.request(destination)
        session.request("human", pause_after_sync=True)
        assert session.action(observation(2), "Pick", lambda: pytest.fail("human read")) is None
        assert session.status == "human paused"
        assert sync.targets == [observation(), observation(2)]
    finally:
        session.close()


@pytest.mark.parametrize("failure", ["read", "write"])
def test_sync_io_failure_never_disables_torque_or_continues_trajectory(monkeypatch, failure):
    now = [0.0]
    monkeypatch.setattr("so101_nexus.teleop.leader.time.monotonic", lambda: now[0])
    bus = Bus()
    sync = SOLeaderSynchronizer(bus, wrist_roll_offset_deg=0)
    sync.start(observation(10))
    bus.positions = dict.fromkeys(SO101_JOINT_NAMES, 3.0)
    method = "sync_read" if failure == "read" else "sync_write"
    monkeypatch.setattr(
        bus, method, lambda *args, **kwargs: (_ for _ in ()).throw(OSError("bus failed"))
    )
    now[0] = 0.1
    with pytest.raises((OSError, RuntimeError), match="bus failed"):
        sync.tick()
    assert ("disable",) not in bus.events
    assert sync.tick()


def test_cancel_holds_current_pose_instead_of_finishing_alignment(monkeypatch):
    now = [0.0]
    monkeypatch.setattr("so101_nexus.teleop.leader.time.monotonic", lambda: now[0])
    bus = Bus()
    sync = SOLeaderSynchronizer(bus, wrist_roll_offset_deg=0)
    sync.start(observation(40))
    bus.positions = dict.fromkeys(SO101_JOINT_NAMES, 5.0)
    sync.cancel()
    assert bus.events[-12:] == [
        event for name in SO101_JOINT_NAMES for event in (("hold", name, 5.0), ("enable", name))
    ]
    assert ("disable",) not in bus.events
    assert sync.tick()


def test_paused_alignment_keeps_support_until_correction_explicitly_starts():
    sync = Synchronizer()
    sync.ready = True
    session = InterventionSession(Policy(), sync)
    try:
        session.begin_episode()
        session.request("human", pause_after_sync=True)
        session.action(observation(), "Pick", lambda: pytest.fail("human read"))
        assert sync.releases == 0
        assert session.start_correction()
        session.action(observation(), "Pick", lambda: observation(3))
        session.action(observation(), "Pick", lambda: observation(3))
        assert sync.releases == 1
        session.end_episode()
        assert sync.releases == 1
    finally:
        session.close()
    assert sync.releases == 1


def test_explicit_idle_release_survives_repeated_cleanup_without_reengaging():
    sync = Synchronizer()
    session = InterventionSession(Policy(), sync)
    session.end_episode()
    assert session.release_leader()
    cancels = sync.cancels
    session.end_episode()
    session.close()
    assert sync.cancels == cancels
    assert sync.releases == 1
    assert session.leader_released


def test_recording_failure_holds_physical_pose_and_preserves_error(worker_setup, monkeypatch):  # noqa: F811
    from so101_nexus.teleop.recorder import RecordingWorker

    kwargs, _, _ = worker_setup
    bus = Bus()
    bus.positions = dict.fromkeys(SO101_JOINT_NAMES, 7.0)
    kwargs["leader"].get_action = lambda: {
        f"{key}.pos": value for key, value in bus.positions.items()
    }
    policy = Policy()

    def fail(batch):
        raise RuntimeError("teleoperation inference failed")

    policy.select_action = fail
    session = InterventionSession(policy, SOLeaderSynchronizer(bus, wrist_roll_offset_deg=0))
    start = session.synchronizer.start
    monkeypatch.setattr(
        session.synchronizer, "start", lambda target, **kwargs: start(target, duration_s=0.001)
    )
    worker = RecordingWorker()
    try:
        worker.submit(**kwargs, intervention=session)
        assert worker.wait(3)
        assert "teleoperation inference failed" in kwargs["state"].error
        assert session.synchronizer.holding
        assert ("disable",) not in bus.events
        assert bus.events[-12:] == [
            event for name in SO101_JOINT_NAMES for event in (("hold", name, 7.0), ("enable", name))
        ]
    finally:
        worker.close()
        session.close()
    assert ("disable",) not in bus.events


def test_pause_arriving_during_correction_start_prevents_motor_release():
    sync = Synchronizer()
    sync.ready = True
    session = InterventionSession(Policy(), sync)
    try:
        session.begin_episode()
        session.request("human", pause_after_sync=True)
        session.action(observation(), "Pick", lambda: pytest.fail("human read"))
        assert session.start_correction()
        cancel = sync.cancel

        def pause_during_hold():
            sync.cancel = cancel
            cancel()
            session.request("hold")

        sync.cancel = pause_during_hold
        assert session.action(observation(), "Pick", lambda: pytest.fail("human read")) is None
        assert sync.releases == 0
    finally:
        session.close()


def test_unused_session_cleanup_does_not_engage_leader_motors():
    sync = Synchronizer()
    session = InterventionSession(Policy(), sync)
    session.close()
    assert sync.cancels == 0
    assert sync.releases == 0


def test_failed_motor_release_invalidates_hold_status(monkeypatch):
    bus = Bus()
    sync = SOLeaderSynchronizer(bus, wrist_roll_offset_deg=0)
    sync.cancel()
    assert sync.holding

    def fail(**kwargs):
        raise OSError("partial torque release")

    monkeypatch.setattr(bus, "disable_torque", fail)
    with pytest.raises(OSError, match="partial torque release"):
        sync.release()
    assert not sync.holding


def test_stale_support_token_cannot_authorize_motor_release():
    sync = Synchronizer()
    sync.ready = True
    session = InterventionSession(Policy(), sync)
    try:
        old = session.support_token
        session.begin_episode()
        session.request("human", pause_after_sync=True)
        session.action(observation(), "Pick", lambda: pytest.fail("human read"))
        assert not session.start_correction(support_token=old)
        session.end_episode()
        assert not session.release_leader(support_token=old)
        assert sync.releases == 0
        assert session.release_leader(support_token=session.support_token)
    finally:
        session.close()


def test_alignment_already_converged_is_accepted_after_deadline(monkeypatch):
    now = [0.0]
    monkeypatch.setattr("so101_nexus.teleop.leader.time.monotonic", lambda: now[0])
    bus = Bus()
    sync = SOLeaderSynchronizer(bus, wrist_roll_offset_deg=0)
    sync.start(observation(10))
    bus.positions = dict.fromkeys(SO101_JOINT_NAMES, 10.0)
    now[0] = 10.0
    assert sync.tick()
    assert sync.holding
    assert ("disable",) not in bus.events


def test_alignment_timeout_reports_joint_errors_without_new_motion(monkeypatch):
    now = [0.0]
    monkeypatch.setattr("so101_nexus.teleop.leader.time.monotonic", lambda: now[0])
    bus = Bus()
    sync = SOLeaderSynchronizer(
        bus, wrist_roll_offset_deg=0, joint_tolerance_deg=2, gripper_tolerance_percent=2
    )
    sync.start(observation(10))
    bus.positions = dict.fromkeys(SO101_JOINT_NAMES, 10.0)
    bus.positions["wrist_flex"] = 6.0
    bus.positions["gripper"] = 5.0
    now[0] = 10.0
    with pytest.raises(TimeoutError, match=r"wrist_flex.*6.0.*10.0.*deg") as error:
        sync.tick()
    assert "gripper" in str(error.value)
    assert "percent" in str(error.value)
    assert bus.events[-12:] == [
        event
        for name in SO101_JOINT_NAMES
        for event in (("hold", name, bus.positions[name]), ("enable", name))
    ]
    assert bus.positions["wrist_flex"] == 6.0
    assert ("disable",) not in bus.events


@pytest.mark.parametrize("destination", ["human", "policy"])
def test_alignment_timeout_pauses_until_explicit_retry_or_policy_resume(destination):
    sync = Synchronizer()
    calls = []

    def tick():
        calls.append(None)
        if len(calls) == 1:
            raise TimeoutError("wrist_flex is 4 deg away")
        return True

    sync.tick = tick
    session = InterventionSession(Policy(), sync)
    try:
        session.begin_episode()
        session.request("human", pause_after_sync=True)
        for _ in range(3):
            assert session.action(observation(), "Pick", lambda: pytest.fail("human read")) is None
            assert session.status == "alignment paused"
            assert session.source == "hold"
            assert "wrist_flex" in session.synchronization_error
        assert len(calls) == 1
        assert session.intervention_id == 0
        session.request(destination, pause_after_sync=True)
        if destination == "policy":
            assert await_action(session, observation()) == observation(10)
        else:
            assert session.action(observation(), "Pick", lambda: pytest.fail("human read")) is None
            assert session.status == "human paused"
            assert len(sync.targets) == 2
        assert session.synchronization_error is None
        assert sync.releases == 0
    finally:
        session.close()


def test_moving_leader_during_correction_never_restarts_alignment():
    sync = Synchronizer()
    sync.ready = True
    session = InterventionSession(Policy(), sync)
    try:
        session.begin_episode()
        session.request("human", pause_after_sync=True)
        session.action(observation(), "Pick", lambda: pytest.fail("human read"))
        assert session.start_correction()
        for value in (3, 4, 20):
            assert session.action(
                observation(), "Pick", lambda value=value: observation(value)
            ) == observation(value - 3)
        assert sync.targets == [observation()]
        assert session.status == "human"
    finally:
        session.close()


def test_takeover_wraps_offset_target_into_physical_wrist_range(monkeypatch):
    from so101_nexus.teleop.leader import apply_wrist_roll_offset_deg

    now = [0.0]
    monkeypatch.setattr("so101_nexus.teleop.leader.time.monotonic", lambda: now[0])
    bus = Bus()
    sync = SOLeaderSynchronizer(bus, wrist_roll_offset_deg=-90)
    follower_pose = {**observation(), "wrist_roll.pos": 163.0}
    sync.start(follower_pose)
    now[0] = 1.0
    assert not sync.tick()
    assert bus.positions["wrist_roll"] == pytest.approx(-20.0, abs=1e-12)
    now[0] = 6.0
    assert sync.tick()
    assert bus.positions["wrist_roll"] == -107.0
    corrected = apply_wrist_roll_offset_deg(
        {f"{name}.pos": value for name, value in bus.positions.items()}, -90.0
    )
    assert corrected == follower_pose


def test_worker_preserves_episode_after_alignment_timeout(worker_setup):  # noqa: F811
    from so101_nexus.teleop.recorder import RecordingWorker

    kwargs, follower_type, _ = worker_setup
    timed_out = threading.Event()
    sync = Synchronizer()

    def tick():
        if sync.parking:
            return True
        timed_out.set()
        raise TimeoutError("wrist_flex is 4 deg away")

    sync.tick = tick
    session = InterventionSession(Policy(), sync)
    original = follower_type.get_observation
    requested = False

    def get_observation(follower):
        nonlocal requested
        if not requested and kwargs["state"].is_recording:
            requested = True
            session.request("human", pause_after_sync=True)
        return original(follower)

    follower_type.get_observation = get_observation
    worker = RecordingWorker()
    try:
        worker.submit(**kwargs, intervention=session)
        assert timed_out.wait(3)
        assert not worker.wait(0.05)
        state = kwargs["state"]
        assert state.is_recording
        assert not state.recording_finished
        assert state.error is None
        assert [t.action for t in state.transitions] == []
        assert [t.intervention_id for t in state.transitions] == []
        session.request("policy")
        assert worker.wait(3)
        assert state.error is None
        assert len(state.transitions) == 1
        assert [t.intervention_id for t in state.transitions] == [0]
    finally:
        worker.close()
        session.close()


def test_wrist_alignment_accepts_equivalent_single_turn_endpoints(monkeypatch):
    now = [0.0]
    monkeypatch.setattr("so101_nexus.teleop.leader.time.monotonic", lambda: now[0])
    bus = Bus()
    sync = SOLeaderSynchronizer(bus, wrist_roll_offset_deg=-90)
    sync.start({**observation(), "wrist_roll.pos": 90.0})
    bus.positions["wrist_roll"] = 179.95
    now[0] = 20.0
    assert sync.tick()
    assert sync.holding
    assert ("disable",) not in bus.events


def test_supported_correction_can_start_from_timed_out_pose_without_a_jump():
    sync = Synchronizer()

    def tick():
        raise TimeoutError("shoulder_lift measured 38.0 target 30.3")

    sync.tick = tick
    session = InterventionSession(Policy(), sync)
    follower = {**observation(), "shoulder_lift.pos": 30.3, "elbow_flex.pos": -36.8}
    leader = {**observation(), "shoulder_lift.pos": 38.0, "elbow_flex.pos": -29.4}
    try:
        session.begin_episode()
        session.request("human", pause_after_sync=True)
        assert session.action(follower, "Pick", lambda: dict(leader)) is None
        assert session.status == "alignment paused"
        assert session.start_correction(support_token=session.support_token)
        first = session.action(follower, "Pick", lambda: dict(leader))
        assert first == follower
        assert sync.releases == 1
        assert len(sync.targets) == 1
        assert session.intervention_id == 1
        leader["shoulder_lift.pos"] += 2.0
        leader["elbow_flex.pos"] -= 1.0
        moved = session.action(follower, "Pick", lambda: dict(leader))
        assert moved["shoulder_lift.pos"] == pytest.approx(32.3)
        assert moved["elbow_flex.pos"] == pytest.approx(-37.8)
        session.request("policy")
        assert await_action(session, follower) == observation(10)
    finally:
        session.close()


def test_invalid_leader_anchor_cannot_release_motors():
    sync = Synchronizer()
    sync.ready = True
    session = InterventionSession(Policy(), sync)
    try:
        session.begin_episode()
        session.request("human", pause_after_sync=True)
        session.action(observation(), "Pick", observation)
        assert session.start_correction()
        with pytest.raises(ValueError, match="finite"):
            session.action(observation(), "Pick", lambda: observation(float("nan")))
        assert sync.releases == 0
    finally:
        session.close()


@pytest.mark.parametrize(
    "field,joint,error,tolerance,ready",
    [
        ("joint_tolerance_deg", "shoulder_lift", 7.7, 10.0, True),
        ("joint_tolerance_deg", "shoulder_lift", 7.7, 2.0, False),
        ("gripper_tolerance_percent", "gripper", 4.0, 5.0, True),
        ("gripper_tolerance_percent", "gripper", 4.0, 2.0, False),
    ],
)
def test_alignment_tolerance_affects_measured_readiness(
    monkeypatch, field, joint, error, tolerance, ready
):
    now = [0.0]
    monkeypatch.setattr("so101_nexus.teleop.leader.time.monotonic", lambda: now[0])
    bus = Bus()
    sync = SOLeaderSynchronizer(bus, wrist_roll_offset_deg=0, **{field: tolerance})
    sync.start(observation(10))
    bus.sync_write = lambda *args, **kwargs: None
    bus.positions = dict.fromkeys(SO101_JOINT_NAMES, 10.0)
    bus.positions[joint] += error
    now[0] = 1.0
    assert sync.tick() is ready


@pytest.mark.parametrize("field", ["joint_tolerance_deg", "gripper_tolerance_percent"])
@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_alignment_tolerances_must_be_positive_and_finite(field, value):
    with pytest.raises(ValueError, match="positive and finite"):
        SOLeaderSynchronizer(Bus(), wrist_roll_offset_deg=0, **{field: value})


def test_anchored_correction_preserves_gripper_and_wrist_continuity():
    sync = Synchronizer()
    sync.ready = True
    session = InterventionSession(Policy(), sync)
    follower = {**observation(10), "wrist_roll.pos": 163.0, "gripper.pos": 60.0}
    leader = {**observation(20), "wrist_roll.pos": 179.9, "gripper.pos": 20.0}
    try:
        session.begin_episode()
        session.request("human", pause_after_sync=True)
        session.action(follower, "Pick", lambda: dict(leader))
        assert session.start_correction()
        assert session.action(follower, "Pick", lambda: dict(leader)) == follower
        leader["wrist_roll.pos"] = -179.9
        leader["gripper.pos"] = 21.0
        action = session.action(follower, "Pick", lambda: dict(leader))
        assert action["wrist_roll.pos"] == pytest.approx(163.2)
        assert action["gripper.pos"] == 61
        session.request("hold")
        session.action(follower, "Pick", lambda: pytest.fail("human read"))
        session.request("human", pause_after_sync=True)
        session.action(follower, "Pick", lambda: pytest.fail("human read"))
        assert session.start_correction()
        assert session.action(follower, "Pick", lambda: dict(leader)) == follower
        assert session.intervention_id == 2
    finally:
        session.close()


def test_failed_hold_cannot_authorize_current_pose_takeover():
    sync = Synchronizer()
    session = InterventionSession(Policy(), sync)
    try:
        session.begin_episode()
        session.request("human", pause_after_sync=True)
        session.action(observation(), "Pick", observation)
        session.status = "alignment paused"
        sync.holding = False
        assert not session.start_correction(support_token=session.support_token)
        assert sync.releases == 0
    finally:
        session.close()


@pytest.mark.parametrize("persistent", [False, True])
def test_pose_read_retries_recover_transient_packets_and_bound_persistent_failures(
    monkeypatch, persistent
):
    pytest.importorskip("scservo_sdk")
    from lerobot.motors.feetech import FeetechMotorsBus

    from so101_nexus.lerobot_adapter.normalization import build_so101_motors

    bus = FeetechMotorsBus(port="/dev/null", motors=build_so101_motors(use_degrees=True))
    monkeypatch.setattr(bus.port_handler, "is_open", True)
    attempts = []

    def read_packet():
        attempts.append(1)
        return -6 if persistent or len(attempts) == 1 else 0

    monkeypatch.setattr(bus.sync_reader, "txRxPacket", read_packet)
    monkeypatch.setattr(bus.sync_reader, "isAvailable", lambda *args: True)
    monkeypatch.setattr(bus.sync_reader, "getData", lambda *args: 2048)
    monkeypatch.setattr(bus, "_normalize", lambda values: dict.fromkeys(values, 0.0))
    holds = []
    monkeypatch.setattr(bus, "write", lambda register, name, value, **kwargs: holds.append(name))
    enables = []
    monkeypatch.setattr(bus, "enable_torque", lambda *, motors, **kwargs: enables.append(motors))
    sync = SOLeaderSynchronizer(bus, wrist_roll_offset_deg=0)
    if persistent:
        with pytest.raises(ConnectionError, match="after 4 tries"):
            sync.cancel()
        assert len(attempts) == 4
        assert not sync.holding
        assert holds == enables == []
    else:
        sync.cancel()
        assert len(attempts) == 2
        assert holds == list(SO101_JOINT_NAMES)
        assert enables == list(SO101_JOINT_NAMES)
        assert sync.holding


def test_hold_requires_acknowledged_targets_before_enabling_torque():
    bus = Bus()
    bus.positions = dict.fromkeys(SO101_JOINT_NAMES, 15.0)
    attempted = []

    def write(register, name, value, *, num_retry=0):
        attempted.append((register, name, value, num_retry))
        if name == SO101_JOINT_NAMES[0]:
            raise ConnectionError("Goal_Position acknowledgment missing")

    bus.write = write
    sync = SOLeaderSynchronizer(bus, wrist_roll_offset_deg=0)
    with pytest.raises(ConnectionError, match="acknowledgment missing"):
        sync.cancel()
    assert attempted == [("Goal_Position", name, 15.0, 3) for name in SO101_JOINT_NAMES]
    assert not sync.holding
    assert ("enable", SO101_JOINT_NAMES[0]) not in bus.events
    assert [event for event in bus.events if event[0] == "enable"] == [
        ("enable", name) for name in SO101_JOINT_NAMES[1:]
    ]
    assert ("disable",) not in bus.events


@pytest.mark.parametrize("persistent", [False, True])
def test_hold_goal_acknowledgments_retry_before_enabling_torque(monkeypatch, persistent):
    pytest.importorskip("scservo_sdk")
    from lerobot.motors.feetech import FeetechMotorsBus

    from so101_nexus.lerobot_adapter.normalization import build_so101_motors

    bus = FeetechMotorsBus(port="/dev/null", motors=build_so101_motors(use_degrees=True))
    monkeypatch.setattr(bus.port_handler, "is_open", True)
    monkeypatch.setattr(
        bus, "sync_read", lambda *args, **kwargs: dict.fromkeys(SO101_JOINT_NAMES, 15.0)
    )
    monkeypatch.setattr(bus, "_unnormalize", lambda values: dict.fromkeys(values, 2048))
    goal_address = bus.model_ctrl_table["sts3215"]["Goal_Position"][0]
    goal_attempts = []
    torque_enabled = []
    monkeypatch.setattr(
        bus, "enable_torque", lambda *, motors, **kwargs: torque_enabled.append(motors)
    )

    def write_packet(_port, motor_id, address, _length, data):
        assert address == goal_address
        assert data == [0, 8]
        goal_attempts.append(motor_id)
        return (-6, 0) if motor_id == 1 and (persistent or goal_attempts.count(1) == 1) else (0, 0)

    monkeypatch.setattr(bus.packet_handler, "writeTxRx", write_packet)
    sync = SOLeaderSynchronizer(bus, wrist_roll_offset_deg=0)
    if persistent:
        with pytest.raises(ConnectionError, match=r"Goal_Position.*after 4 tries"):
            sync.cancel()
        assert goal_attempts == [1, 1, 1, 1, 2, 3, 4, 5, 6]
        assert torque_enabled == list(SO101_JOINT_NAMES[1:])
        assert not sync.holding
    else:
        sync.cancel()
        assert goal_attempts == [1, 1, 2, 3, 4, 5, 6]
        assert torque_enabled == list(SO101_JOINT_NAMES)
        assert sync.holding


@pytest.mark.parametrize("persistent", [False, True])
def test_alignment_transmission_retries_are_bounded(monkeypatch, persistent):
    pytest.importorskip("scservo_sdk")
    from lerobot.motors.feetech import FeetechMotorsBus

    from so101_nexus.lerobot_adapter.normalization import build_so101_motors

    bus = FeetechMotorsBus(port="/dev/null", motors=build_so101_motors(use_degrees=True))
    monkeypatch.setattr(bus.port_handler, "is_open", True)
    monkeypatch.setattr(bus, "_unnormalize", lambda values: dict.fromkeys(values, 2048))
    monkeypatch.setattr(
        bus, "sync_read", lambda *args, **kwargs: dict.fromkeys(SO101_JOINT_NAMES, 0.0)
    )
    attempts = []

    def write_packet():
        attempts.append(1)
        return -6 if persistent or len(attempts) == 1 else 0

    monkeypatch.setattr(bus.sync_writer, "txPacket", write_packet)
    sync = SOLeaderSynchronizer(bus, wrist_roll_offset_deg=0)
    sync._initial = dict.fromkeys(SO101_JOINT_NAMES, 0.0)
    monkeypatch.setattr(sync, "cancel", lambda: None)
    sync.start(observation(10))
    if persistent:
        with pytest.raises(ConnectionError, match=r"Goal_Position.*after 4 tries"):
            sync.tick()
        assert len(attempts) == 4
    else:
        assert not sync.tick()
        assert len(attempts) == 2


@pytest.mark.parametrize(
    "initial,target,expected", [(179.0, -179.0, 179.0), (160.0, -160.0, 150.0)]
)
def test_wrist_alignment_keeps_equivalent_seam_pose_without_skipping_larger_errors(
    monkeypatch, initial, target, expected
):
    now = [0.0]
    monkeypatch.setattr("so101_nexus.teleop.leader.time.monotonic", lambda: now[0])
    bus = Bus()
    bus.positions["wrist_roll"] = initial
    sync = SOLeaderSynchronizer(bus, wrist_roll_offset_deg=0)
    sync.start({**observation(), "shoulder_lift.pos": 40.0, "wrist_roll.pos": target})
    now[0] = 0.5
    assert not sync.tick()
    assert bus.positions["wrist_roll"] == expected
    assert -180 <= bus.positions["wrist_roll"] <= 180
    assert 0 < bus.positions["shoulder_lift"] <= 10.0
    assert ("disable",) not in bus.events
