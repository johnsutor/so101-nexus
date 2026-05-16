"""Tests for teleop recorder pure helpers and follower-driven loop."""

from __future__ import annotations

import inspect
import io
import os
import threading

import numpy as np
import pytest

from so101_nexus_core.teleop.recorder import (
    RecordingState,
    TeeStream,
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
    from so101_nexus_core.teleop.recorder import PREVIEW_MAX_DIM as exposed_dim

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


def test_publish_camera_frames_extracts_follower_camera_keys() -> None:
    state = RecordingState()
    wrist = np.full((8, 10, 3), 128, dtype=np.uint8)
    overhead = np.full((6, 12, 3), 64, dtype=np.uint8)

    _publish_camera_frames(state, {"wrist": wrist, "overhead": overhead})

    assert state.episode_wrist_images == [wrist]
    assert state.episode_overhead_images == [overhead]
    assert state.live_preview is not None


def test_publish_camera_frames_extracts_maniskill_sensor_data() -> None:
    state = RecordingState()
    obs = {
        "sensor_data": {
            "wrist_camera": {"rgb": np.full((1, 8, 10, 3), 128, dtype=np.uint8)},
            "overhead_camera": {"rgb": np.full((1, 6, 12, 3), 64, dtype=np.uint8)},
        }
    }

    _publish_camera_frames(state, obs)

    assert len(state.episode_wrist_images) == 1
    assert state.episode_wrist_images[0].shape == (8, 10, 3)
    assert len(state.episode_overhead_images) == 1
    assert state.episode_overhead_images[0].shape == (6, 12, 3)
    assert state.live_preview is not None


def test_recording_thread_drives_follower_and_records_state_from_obs(tmp_path) -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("so101_nexus_mujoco")

    from so101_nexus_core.config import SO101_JOINT_NAMES

    state = RecordingState(num_episodes=1)
    leader = _FakeLeader()
    leader.connect()

    thread = threading.Thread(
        target=recording_thread,
        kwargs={
            "state": state,
            "env_id": "MuJoCoReach-v1",
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
    assert not np.array_equal(action0, state0)
    assert state.live_frame is not None
    assert state.live_frame.mean() > 1.0
    assert state.live_preview is not None


def test_recording_thread_records_error_on_leader_failure(tmp_path) -> None:
    pytest.importorskip("so101_nexus_mujoco")

    from so101_nexus_core.config import SO101_JOINT_NAMES

    state = RecordingState(num_episodes=1)
    leader = _ExplodingLeader()
    thread = threading.Thread(
        target=recording_thread,
        kwargs={
            "state": state,
            "env_id": "MuJoCoReach-v1",
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
