"""Tests for teleop recorder pure helpers."""

from __future__ import annotations

import inspect
import io

import numpy as np

from so101_nexus_core.teleop.recorder import (
    RecordingState,
    TeeStream,
    compute_delta_actions,
    recording_thread,
)


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
