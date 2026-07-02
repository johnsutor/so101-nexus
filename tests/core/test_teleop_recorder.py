"""Tests for teleop recorder pure helpers and follower-driven loop."""

from __future__ import annotations

import inspect
import io
import os
import threading
import types

import numpy as np
import pytest

from so101_nexus.config import SO101_JOINT_NAMES
from so101_nexus.teleop.recorder import (
    RecordingState,
    TeeStream,
    _append_step_buffers,
    _publish_camera_frames,
    compute_delta_actions,
    recording_thread,
)

os.environ.setdefault("MUJOCO_GL", "egl")

PREVIEW_MAX_DIM = 320


class _FakeLeader:
    """Deterministic leader returning degree body joints and percent gripper."""

    def __init__(self) -> None:
        self.connected = False

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def get_action(self) -> dict[str, float]:
        return {
            "shoulder_pan.pos": 0.0,
            "shoulder_lift.pos": 0.0,
            "elbow_flex.pos": 0.0,
            "wrist_flex.pos": 0.0,
            "wrist_roll.pos": 0.0,
            "gripper.pos": 50.0,
        }


class _ExplodingLeader(_FakeLeader):
    def get_action(self) -> dict[str, float]:
        raise RuntimeError("simulated leader read failure")


class _FakeRecordingFollower:
    terminated_after_step = 10**9
    instances: list[_FakeRecordingFollower] = []

    def __init__(self, _config) -> None:
        self.step_calls = 0
        self.initial_leader_action: dict[str, float] | None = None
        self._last_step_info = None
        self._env = types.SimpleNamespace(
            unwrapped=types.SimpleNamespace(task_description="Fake recording task")
        )
        type(self).instances.append(self)

    def set_initial_leader_action(self, action: dict[str, float]) -> None:
        self.initial_leader_action = dict(action)

    def connect(self) -> None:
        pass

    def disconnect(self) -> None:
        pass

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        self.step_calls += 1
        terminated = self.step_calls >= type(self).terminated_after_step
        self._last_step_info = types.SimpleNamespace(
            terminated=terminated,
            truncated=False,
            info={"success": terminated},
            reward=float(terminated),
        )
        return dict(action)

    def get_observation(self) -> dict[str, object]:
        obs = {f"{name}.pos": float(self.step_calls) for name in SO101_JOINT_NAMES}
        obs["wrist"] = np.zeros((4, 4, 3), dtype=np.uint8)
        return obs

    def last_step_info(self):
        return self._last_step_info


def _run_fake_recording(
    monkeypatch,
    *,
    terminated_after_step: int = 10**9,
    max_steps: int = 10,
    fps: int = 30,
    success_hold_seconds: float = 0.5,
    leader=None,
) -> tuple[RecordingState, _FakeRecordingFollower]:
    import so101_nexus.lerobot_adapter.sim_follower as sim_follower_module
    import so101_nexus.teleop.recorder as recorder_module
    import so101_nexus.teleop.session as session_module

    _FakeRecordingFollower.instances = []
    _FakeRecordingFollower.terminated_after_step = terminated_after_step
    monkeypatch.setattr(sim_follower_module, "SimSOFollower", _FakeRecordingFollower)
    monkeypatch.setattr(session_module, "prepare_follower_calibration", lambda **_kwargs: None)
    monkeypatch.setattr(session_module, "build_sim_follower_config", lambda **_kwargs: object())
    monkeypatch.setattr(recorder_module.time, "sleep", lambda _seconds: None)

    state = RecordingState(num_episodes=1)
    recording_thread(
        state=state,
        env_id="FakeEnv-v0",
        leader=leader or _FakeLeader(),
        joint_names=SO101_JOINT_NAMES,
        fps=fps,
        max_steps=max_steps,
        countdown=0,
        wrist_roll_offset_deg=-90.0,
        wrist_wh=(4, 4),
        overhead_wh=(4, 4),
        follower_calibration_dir="unused",
        follower_robot_id="teleop_sim_test",
        success_hold_seconds=success_hold_seconds,
    )
    return state, _FakeRecordingFollower.instances[-1]


def test_compute_delta_actions_zero_prefix() -> None:
    a0 = np.array([1.0, 2.0, 3.0])
    a1 = np.array([1.5, 2.5, 3.5])
    a2 = np.array([2.0, 3.0, 4.0])
    deltas = compute_delta_actions([a0, a1, a2])
    assert len(deltas) == 3
    np.testing.assert_array_equal(deltas[0], np.zeros(3))
    np.testing.assert_array_equal(deltas[1], a1 - a0)
    np.testing.assert_array_equal(deltas[2], a2 - a1)


def test_recording_state_starts_clean() -> None:
    state = RecordingState()
    assert state.episode_actions == []
    assert state.episode_states == []
    assert state.episode_wrist_images == []
    assert state.episode_overhead_images == []
    assert not state.is_recording


def test_recording_state_exposes_live_preview() -> None:
    from so101_nexus.teleop.recorder import PREVIEW_MAX_DIM as exposed_dim

    assert exposed_dim == PREVIEW_MAX_DIM
    state = RecordingState()
    assert state.live_preview is None
    state.live_preview = np.zeros((10, 10, 3), dtype=np.uint8)
    state.clear_episode()
    assert state.live_preview is None


def test_tee_stream_duplicates_writes() -> None:
    sink = io.StringIO()
    tee = TeeStream(sink)
    tee.write("hello")
    assert sink.getvalue() == "hello"
    assert tee.get_output() == "hello"


def test_recording_thread_signature_drops_action_pipeline_kwarg() -> None:
    sig = inspect.signature(recording_thread)

    assert "action_pipeline" not in sig.parameters
    assert "follower_calibration_dir" in sig.parameters
    assert "follower_robot_id" in sig.parameters
    assert "success_hold_seconds" in sig.parameters


def test_publish_camera_frames_extracts_follower_camera_keys() -> None:
    state = RecordingState()
    wrist = np.full((8, 10, 3), 128, dtype=np.uint8)
    overhead = np.full((6, 12, 3), 64, dtype=np.uint8)

    _publish_camera_frames(state, {"wrist": wrist, "overhead": overhead})

    assert state.episode_wrist_images == [wrist]
    assert state.episode_overhead_images == [overhead]
    assert state.live_preview is not None


def test_recording_thread_stops_after_terminated_plus_hold(monkeypatch) -> None:
    state, _follower = _run_fake_recording(
        monkeypatch,
        terminated_after_step=5,
        max_steps=100,
        fps=10,
        success_hold_seconds=0.5,
    )

    assert state.error is None
    assert state.terminated_at_frame == 5
    assert len(state.episode_actions) == 10


def test_recording_thread_stops_immediately_with_zero_hold(monkeypatch) -> None:
    state, _follower = _run_fake_recording(
        monkeypatch,
        terminated_after_step=3,
        max_steps=100,
        fps=10,
        success_hold_seconds=0.0,
    )

    assert state.error is None
    assert state.terminated_at_frame == 3
    assert len(state.episode_actions) == 3


def test_recording_thread_seeds_follower_with_leader_pose(monkeypatch) -> None:
    state, follower = _run_fake_recording(monkeypatch, max_steps=2)

    assert state.error is None
    assert follower.initial_leader_action is not None
    assert all(key.endswith(".pos") for key in follower.initial_leader_action)
    assert follower.initial_leader_action["wrist_roll.pos"] == -90.0


def test_recording_thread_drives_follower_and_records_state_from_obs(tmp_path) -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("so101_nexus.mujoco")

    from so101_nexus.config import SO101_JOINT_NAMES

    state = RecordingState(num_episodes=1)
    leader = _FakeLeader()
    leader.connect()

    thread = threading.Thread(
        target=recording_thread,
        kwargs={
            "state": state,
            "env_id": "MuJoCoTouch-v1",
            "leader": leader,
            "joint_names": SO101_JOINT_NAMES,
            "fps": 30,
            "max_steps": 5,
            "countdown": 0,
            "wrist_roll_offset_deg": -90.0,
            "wrist_wh": (240, 180),
            "overhead_wh": (320, 180),
            "follower_calibration_dir": tmp_path,
            "follower_robot_id": "teleop_sim_test",
        },
        daemon=True,
    )
    thread.start()
    thread.join(timeout=30)

    assert not thread.is_alive(), "recording thread did not finish"
    assert state.recording_finished
    assert state.error is None, f"recording errored: {state.error}"
    assert len(state.episode_actions) > 0
    assert len(state.episode_states) == len(state.episode_actions)

    action0 = state.episode_actions[0]
    state0 = state.episode_states[0]
    assert action0.shape == (len(SO101_JOINT_NAMES),)
    assert state0.shape == (len(SO101_JOINT_NAMES),)
    assert action0.dtype == np.float32
    assert state0.dtype == np.float32

    gripper_idx = SO101_JOINT_NAMES.index("gripper")
    assert 0.0 <= action0[gripper_idx] <= 100.0
    assert 0.0 <= state0[gripper_idx] <= 100.0
    np.testing.assert_allclose(state0, action0, atol=0.1)
    assert state.live_frame is not None
    assert state.live_frame.mean() > 1.0
    assert state.live_preview is not None


def test_recording_thread_records_error_on_leader_failure(tmp_path) -> None:
    pytest.importorskip("so101_nexus.mujoco")

    from so101_nexus.config import SO101_JOINT_NAMES

    state = RecordingState(num_episodes=1)
    leader = _ExplodingLeader()
    thread = threading.Thread(
        target=recording_thread,
        kwargs={
            "state": state,
            "env_id": "MuJoCoTouch-v1",
            "leader": leader,
            "joint_names": SO101_JOINT_NAMES,
            "fps": 30,
            "max_steps": 5,
            "countdown": 0,
            "wrist_roll_offset_deg": -90.0,
            "wrist_wh": (240, 180),
            "overhead_wh": (320, 180),
            "follower_calibration_dir": tmp_path,
            "follower_robot_id": "teleop_sim_test",
        },
        daemon=True,
    )
    thread.start()
    thread.join(timeout=15)

    assert state.recording_finished
    assert state.error is not None
    assert "simulated leader read failure" in state.error


def _motor_obs(value: float = 0.0) -> dict[str, object]:
    return {f"{name}.pos": value for name in SO101_JOINT_NAMES}


def _step_info(*, terminated=False, truncated=False, info=None, reward=0.0):
    return types.SimpleNamespace(
        terminated=terminated,
        truncated=truncated,
        info={} if info is None else info,
        reward=reward,
    )


def test_append_step_buffers_non_terminal_step_records_zero_success_and_done() -> None:
    state = RecordingState()

    _append_step_buffers(
        state,
        _motor_obs(1.0),
        _motor_obs(2.0),
        _step_info(terminated=False, truncated=False, info={}, reward=0.5),
        SO101_JOINT_NAMES,
    )

    assert state.episode_rewards == [0.5]
    assert state.episode_successes == [0.0]
    assert state.episode_dones == [0.0]


def test_append_step_buffers_terminal_success_records_success_and_done() -> None:
    state = RecordingState()

    _append_step_buffers(
        state,
        _motor_obs(1.0),
        _motor_obs(2.0),
        _step_info(terminated=True, info={"success": True}, reward=1.0),
        SO101_JOINT_NAMES,
    )

    assert state.episode_successes == [1.0]
    assert state.episode_dones == [1.0]


def test_append_step_buffers_truncated_step_records_done_without_success() -> None:
    state = RecordingState()

    _append_step_buffers(
        state,
        _motor_obs(1.0),
        _motor_obs(2.0),
        _step_info(terminated=False, truncated=True, info={}),
        SO101_JOINT_NAMES,
    )

    assert state.episode_successes == [0.0]
    assert state.episode_dones == [1.0]


def test_append_step_buffers_appends_env_state_from_obs() -> None:
    state = RecordingState()
    obs = _motor_obs(2.0)
    obs["environment_state"] = np.arange(18, dtype=np.float64)

    _append_step_buffers(
        state,
        _motor_obs(1.0),
        obs,
        _step_info(),
        SO101_JOINT_NAMES,
    )

    assert len(state.episode_env_states) == 1
    stored = state.episode_env_states[0]
    assert stored.dtype == np.float32
    assert stored.shape == (18,)
    np.testing.assert_array_equal(stored, np.arange(18, dtype=np.float32))


def test_append_step_buffers_pre_first_step_defaults_and_skips_env_state() -> None:
    state = RecordingState()

    # step_info is None before the first env.step (seed frame); obs has no
    # environment_state key, so no privileged vector is buffered.
    _append_step_buffers(
        state,
        _motor_obs(1.0),
        _motor_obs(1.0),
        None,
        SO101_JOINT_NAMES,
    )

    assert state.episode_rewards == [0.0]
    assert state.episode_successes == [0.0]
    assert state.episode_dones == [0.0]
    assert state.episode_env_states == []
