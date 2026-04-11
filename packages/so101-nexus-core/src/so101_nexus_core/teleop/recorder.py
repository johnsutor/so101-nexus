"""Shared recording-thread state and pure helpers for teleop.

Heavy dependencies (``gymnasium``, ``cv2``) are imported lazily inside
:func:`recording_thread` so that this module can be imported and unit-tested
on a base install without the ``teleop`` extra.
"""

from __future__ import annotations

import io
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from so101_nexus_core.teleop.leader import LeaderProtocol


class _WritableTextStream(Protocol):
    """Protocol for text-like streams used by :class:`TeeStream`."""

    def write(self, s: str) -> object: ...

    def flush(self) -> object: ...


class TeeStream(io.TextIOBase):
    """Writable stream that duplicates writes to *original* and an internal buffer."""

    def __init__(self, original: _WritableTextStream) -> None:
        self._original = original
        self._buffer = io.StringIO()
        self._lock = threading.Lock()

    def write(self, s: str) -> int:
        """Write *s* to both the original stream and the internal buffer."""
        with self._lock:
            self._original.write(s)
            self._buffer.write(s)
        return len(s)

    def flush(self) -> None:
        """Flush the original stream."""
        self._original.flush()

    def get_output(self) -> str:
        """Return all buffered output accumulated so far."""
        with self._lock:
            return self._buffer.getvalue()


@dataclass
class RecordingState:
    """Mutable state shared between the recording thread and the Gradio UI."""

    is_recording: bool = False
    should_stop: bool = False
    countdown_value: int = 0
    recording_finished: bool = False

    episode_actions: list[np.ndarray] = field(default_factory=list)
    episode_states: list[np.ndarray] = field(default_factory=list)
    episode_wrist_images: list[np.ndarray] = field(default_factory=list)
    episode_overhead_images: list[np.ndarray] = field(default_factory=list)
    task_description: str = ""
    episode_duration: float = 0.0
    live_frame: np.ndarray | None = None

    episodes_completed: int = 0
    num_episodes: int = 0

    def clear_episode(self) -> None:
        """Reset all per-episode buffers."""
        self.episode_actions.clear()
        self.episode_states.clear()
        self.episode_wrist_images.clear()
        self.episode_overhead_images.clear()
        self.task_description = ""
        self.episode_duration = 0.0
        self.live_frame = None
        self.recording_finished = False


def convert_leader_action(
    action: dict,
    joint_names: tuple[str, ...],
    wrist_roll_offset_deg: float,
) -> np.ndarray:
    """Convert leader arm joint readings (degrees) to radians."""
    converted: list[float] = []
    wrist_roll_offset_rad = np.deg2rad(wrist_roll_offset_deg)
    for name in joint_names:
        value = np.deg2rad(action[f"{name}.pos"])
        if name == "wrist_roll":
            value += wrist_roll_offset_rad
        converted.append(value)
    return np.array(converted, dtype=np.float64)


def compute_delta_actions(actions: list[np.ndarray]) -> list[np.ndarray]:
    """Convert absolute joint positions to frame-to-frame deltas."""
    deltas: list[np.ndarray] = [np.zeros_like(actions[0])]
    for i in range(1, len(actions)):
        deltas.append(actions[i] - actions[i - 1])
    return deltas


def recording_thread(
    state: RecordingState,
    env_id: str,
    leader: LeaderProtocol,
    joint_names: tuple[str, ...],
    fps: int,
    max_steps: int,
    countdown: int,
    wrist_roll_offset_deg: float,
    wrist_wh: tuple[int, int],
    overhead_wh: tuple[int, int],
) -> None:
    """Run a countdown, then record at *fps* until stopped or max_steps reached."""
    import gymnasium as gym

    from so101_nexus_core.teleop.session import _recording_env_kwargs

    for i in range(countdown, 0, -1):
        state.countdown_value = i
        time.sleep(1.0)
    state.countdown_value = 0

    env = gym.make(
        env_id,
        render_mode="rgb_array",
        **_recording_env_kwargs(env_id, wrist_wh, overhead_wh),
    )
    try:
        leader_action = leader.get_action()
        init_qpos = convert_leader_action(
            leader_action,
            joint_names,
            wrist_roll_offset_deg=wrist_roll_offset_deg,
        )
        obs, _ = env.reset(options={"init_qpos": init_qpos})
        state.clear_episode()
        state.task_description = getattr(env.unwrapped, "task_description", "")
        state.is_recording = True

        frame_duration = 1.0 / fps
        start_time = time.monotonic()

        while not state.should_stop:
            step_start = time.monotonic()
            if len(state.episode_actions) >= max_steps:
                break

            leader_action = leader.get_action()
            action = convert_leader_action(
                leader_action,
                joint_names,
                wrist_roll_offset_deg=wrist_roll_offset_deg,
            )
            obs, _, terminated, truncated, _ = env.step(action)

            wrist_image = None
            overhead_image = None
            if isinstance(obs, dict):
                wrist_image = obs.get("wrist_camera")
                overhead_image = obs.get("overhead_camera")

            # The leader arm action IS the observation.state for the dataset —
            # it matches what real robot joint encoders would report.
            state.episode_actions.append(action.astype(np.float32))
            state.episode_states.append(action.astype(np.float32))
            if wrist_image is not None:
                state.episode_wrist_images.append(wrist_image)
                state.live_frame = wrist_image.copy()
            if overhead_image is not None:
                state.episode_overhead_images.append(overhead_image)

            if terminated or truncated:
                break

            sleep_time = frame_duration - (time.monotonic() - step_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

        state.episode_duration = time.monotonic() - start_time
    finally:
        env.close()
        state.is_recording = False
        state.should_stop = False
        state.recording_finished = True
