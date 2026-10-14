"""Tests for policy collection with interruptible human control."""

from __future__ import annotations

import threading
import time

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
        self.ready = False

    def start(self, target):
        self.targets.append(target)

    def tick(self):
        return self.ready

    def cancel(self):
        self.cancels += 1


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

    def sync_read(self, register):
        return dict(self.positions)

    def sync_write(self, register, values):
        self.events.append(("write", dict(values)))
        self.positions.update(values)

    def enable_torque(self):
        self.events.append(("enable",))

    def disable_torque(self):
        self.events.append(("disable",))


def test_leader_sync_interpolates_converts_offset_and_releases_torque(monkeypatch):
    now = [0.0]
    monkeypatch.setattr("so101_nexus.teleop.leader.time.monotonic", lambda: now[0])
    bus = Bus()
    sync = SOLeaderSynchronizer(bus, wrist_roll_offset_deg=-90)
    target = observation(10)
    sync.start(target)
    assert bus.events[:2] == [("write", dict.fromkeys(SO101_JOINT_NAMES, 0.0)), ("enable",)]
    assert not sync.tick()
    now[0] = 1.0
    assert not sync.tick()
    assert bus.positions["wrist_roll"] == 20
    now[0] = 5.0
    assert sync.tick()
    assert bus.positions["wrist_roll"] == 100
    assert bus.positions["gripper"] == 10
    assert bus.events[-1] == ("disable",)


def test_leader_sync_timeout_and_cancel_disable_torque(monkeypatch):
    now = [0.0]
    monkeypatch.setattr("so101_nexus.teleop.leader.time.monotonic", lambda: now[0])
    bus = Bus()
    sync = SOLeaderSynchronizer(bus, wrist_roll_offset_deg=0)
    sync.start(observation(10))
    bus.sync_write = lambda *args: None
    now[0] = 10
    with pytest.raises(TimeoutError, match="synchron"):
        sync.tick()
    assert bus.events[-1] == ("disable",)
    sync.cancel()
    assert bus.events[-1] == ("disable",)


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
        np.testing.assert_array_equal(state.episode_actions[0], np.full(6, 10))
        assert state.episode_intervention_ids == [0]
        assert session.status == "hold"
    finally:
        worker.close()
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
        assert state.episode_intervention_ids == [0, 1, 0]
        assert len(state.episode_wrist_images) == len(state.episode_actions) == 3
        np.testing.assert_array_equal(state.episode_states, [np.full(6, i) for i in range(3)])
        assert sync.targets == [observation(1)]
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
