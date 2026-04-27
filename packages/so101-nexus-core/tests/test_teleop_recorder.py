"""Tests for teleop recorder pure helpers."""

from __future__ import annotations

import io
import threading

import numpy as np

from so101_nexus_core.config import SO101_JOINT_NAMES
from so101_nexus_core.teleop.recorder import (
    RecordingState,
    TeeStream,
    compute_delta_actions,
    convert_leader_action,
)


def test_compute_delta_actions_zero_prefix() -> None:
    a0 = np.array([1.0, 2.0, 3.0])
    a1 = np.array([1.5, 2.5, 3.5])
    a2 = np.array([2.0, 2.0, 4.0])
    deltas = compute_delta_actions([a0, a1, a2])

    assert len(deltas) == 3
    np.testing.assert_allclose(deltas[0], np.zeros(3))
    np.testing.assert_allclose(deltas[1], [0.5, 0.5, 0.5])
    np.testing.assert_allclose(deltas[2], [0.5, -0.5, 0.5])


def test_convert_leader_action_degrees_to_radians() -> None:
    action = {f"{name}.pos": 0.0 for name in SO101_JOINT_NAMES}
    action["shoulder_pan.pos"] = 90.0
    action["wrist_roll.pos"] = 0.0

    out = convert_leader_action(
        action,
        SO101_JOINT_NAMES,
        wrist_roll_offset_deg=0.0,
    )

    assert out.shape == (len(SO101_JOINT_NAMES),)
    np.testing.assert_allclose(out[0], np.pi / 2)


def test_recording_state_clear_episode_resets_overhead_frame() -> None:
    state = RecordingState()
    state.live_overhead_frame = np.zeros((64, 64, 3), dtype=np.uint8)
    state.live_frame = np.zeros((64, 64, 3), dtype=np.uint8)
    state.episode_overhead_images.append(np.zeros((64, 64, 3), dtype=np.uint8))

    state.clear_episode()

    assert state.live_overhead_frame is None
    assert state.live_frame is None
    assert len(state.episode_overhead_images) == 0


def test_convert_leader_action_applies_wrist_roll_offset_only_to_wrist_roll() -> None:
    action = {f"{name}.pos": 0.0 for name in SO101_JOINT_NAMES}

    out = convert_leader_action(
        action,
        SO101_JOINT_NAMES,
        wrist_roll_offset_deg=-90.0,
    )

    wrist_roll_idx = SO101_JOINT_NAMES.index("wrist_roll")
    for i, value in enumerate(out):
        if i == wrist_roll_idx:
            np.testing.assert_allclose(value, -np.pi / 2)
        else:
            np.testing.assert_allclose(value, 0.0)


def test_tee_stream_write_duplicates_to_original_and_buffer() -> None:
    original = io.StringIO()
    tee = TeeStream(original)

    n = tee.write("hello")

    assert n == len("hello")
    assert original.getvalue() == "hello"
    assert tee.get_output() == "hello"


def test_tee_stream_flush_delegates_to_original() -> None:
    calls = {"flush": 0}

    class _StubStream:
        def write(self, s: str) -> int:
            return len(s)

        def flush(self) -> None:
            calls["flush"] += 1

    tee = TeeStream(_StubStream())
    tee.flush()

    assert calls["flush"] == 1


def test_tee_stream_get_output_accumulates_across_writes() -> None:
    tee = TeeStream(io.StringIO())

    tee.write("foo")
    tee.write("bar")
    tee.write("baz")

    assert tee.get_output() == "foobarbaz"


def test_tee_stream_concurrent_writes_preserve_total_length() -> None:
    tee = TeeStream(io.StringIO())
    payload = "abcdefgh"
    n_threads = 8
    writes_per_thread = 100

    def worker() -> None:
        for _ in range(writes_per_thread):
            tee.write(payload)

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    expected = len(payload) * n_threads * writes_per_thread
    assert len(tee.get_output()) == expected
