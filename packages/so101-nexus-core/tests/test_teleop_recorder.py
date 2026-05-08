"""Tests for teleop recorder pure helpers."""

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
    compute_delta_actions,
    recording_thread,
)

os.environ.setdefault("MUJOCO_GL", "egl")

PREVIEW_MAX_DIM = 320


class _FakeLeader:
    """Deterministic leader that returns a fixed joint dict in degrees."""

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
            "gripper.pos": 0.0,
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
    """`live_preview` is a single composite numpy frame for the UI to render."""
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


def test_recording_thread_accepts_optional_action_pipeline_kwarg() -> None:
    """Smoke test that the public function signature accepts the new kwarg."""
    sig = inspect.signature(recording_thread)
    assert "action_pipeline" in sig.parameters
    assert sig.parameters["action_pipeline"].default is None


def test_recording_thread_populates_live_preview() -> None:
    """The recording loop must publish at least one non-black wrist frame."""
    pytest.importorskip("mujoco")
    pytest.importorskip("so101_nexus_mujoco")

    from so101_nexus_core.config import SO101_JOINT_NAMES

    state = RecordingState(num_episodes=1)
    leader = _FakeLeader()
    leader.connect()

    thread = threading.Thread(
        target=recording_thread,
        args=(
            state,
            "MuJoCoReach-v1",
            leader,
            SO101_JOINT_NAMES,
            30,
            5,
            0,
            -90.0,
            (240, 180),
            (320, 180),
        ),
        daemon=True,
    )
    thread.start()
    thread.join(timeout=20)
    assert not thread.is_alive(), "recording thread did not finish"
    assert state.recording_finished, "recording_finished flag was not set"
    assert state.live_frame is not None, "live_frame was never populated"
    assert state.live_frame.mean() > 1.0, "live_frame was all-black"
    assert state.live_preview is not None, "live_preview was never populated"
    assert state.error is None, f"recording errored: {state.error}"


def test_recording_thread_records_error_on_leader_failure() -> None:
    pytest.importorskip("so101_nexus_mujoco")

    from so101_nexus_core.config import SO101_JOINT_NAMES

    state = RecordingState(num_episodes=1)
    leader = _ExplodingLeader()
    thread = threading.Thread(
        target=recording_thread,
        args=(
            state,
            "MuJoCoReach-v1",
            leader,
            SO101_JOINT_NAMES,
            30,
            5,
            0,
            -90.0,
            (240, 180),
            (320, 180),
        ),
        daemon=True,
    )
    thread.start()
    thread.join(timeout=8)
    assert state.recording_finished
    assert state.error is not None
    assert "simulated leader read failure" in state.error
