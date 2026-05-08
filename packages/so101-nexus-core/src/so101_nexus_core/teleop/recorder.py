"""Shared recording-thread state and pure helpers for teleop.

Heavy dependencies (``gymnasium``, ``cv2``) are imported lazily inside
:func:`recording_thread` so that this module can be imported and unit-tested
on a base install without the ``teleop`` extra.
"""

from __future__ import annotations

import io
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np

if TYPE_CHECKING:
    from lerobot.processor import DataProcessorPipeline

    from so101_nexus_core.teleop.leader import LeaderProtocol


PREVIEW_MAX_DIM = 320


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
    error: str | None = None

    episode_actions: list[np.ndarray] = field(default_factory=list)
    episode_states: list[np.ndarray] = field(default_factory=list)
    episode_wrist_images: list[np.ndarray] = field(default_factory=list)
    episode_overhead_images: list[np.ndarray] = field(default_factory=list)
    task_description: str = ""
    episode_duration: float = 0.0
    live_frame: np.ndarray | None = None
    live_overhead_frame: np.ndarray | None = None
    live_preview: np.ndarray | None = None

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
        self.live_overhead_frame = None
        self.live_preview = None
        self.recording_finished = False
        self.error = None


def compute_delta_actions(actions: list[np.ndarray]) -> list[np.ndarray]:
    """Convert absolute joint positions to frame-to-frame deltas."""
    deltas: list[np.ndarray] = [np.zeros_like(actions[0])]
    for i in range(1, len(actions)):
        deltas.append(actions[i] - actions[i - 1])
    return deltas


def _make_preview_frame(
    wrist: np.ndarray | None,
    overhead: np.ndarray | None,
    max_dim: int = PREVIEW_MAX_DIM,
) -> np.ndarray | None:
    """Combine wrist and overhead frames into one downscaled side-by-side preview."""
    import cv2

    def _fit(img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        scale = max_dim / max(h, w)
        if scale >= 1.0:
            return img
        return cv2.resize(
            img,
            (round(w * scale), round(h * scale)),
            interpolation=cv2.INTER_AREA,
        )

    tiles: list[np.ndarray] = []
    if wrist is not None:
        tiles.append(_fit(wrist))
    if overhead is not None:
        tiles.append(_fit(overhead))
    if not tiles:
        return None

    target_h = max(tile.shape[0] for tile in tiles)
    padded: list[np.ndarray] = []
    for tile in tiles:
        pad = target_h - tile.shape[0]
        padded_tile = np.pad(tile, ((0, pad), (0, 0), (0, 0))) if pad else tile
        padded.append(padded_tile)

    out = padded[0]
    gutter = np.zeros((target_h, 4, 3), dtype=out.dtype)
    for tile in padded[1:]:
        out = np.concatenate([out, gutter, tile], axis=1)
    return out


def _publish_camera_frames(state: RecordingState, obs: object) -> None:
    """Copy camera observations into episode buffers and the UI preview slot."""
    wrist_image = None
    overhead_image = None
    if isinstance(obs, Mapping):
        camera_obs = cast("Mapping[str, np.ndarray]", obs)
        wrist_image = camera_obs.get("wrist_camera")
        overhead_image = camera_obs.get("overhead_camera")

    if wrist_image is not None:
        state.episode_wrist_images.append(wrist_image)
        state.live_frame = wrist_image
    if overhead_image is not None:
        state.episode_overhead_images.append(overhead_image)
        state.live_overhead_frame = overhead_image
    state.live_preview = _make_preview_frame(wrist_image, overhead_image)


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
    action_pipeline: DataProcessorPipeline | None = None,
) -> None:
    """Run a countdown, then record at *fps* until stopped or max_steps reached.

    Parameters
    ----------
    action_pipeline
        Optional ``DataProcessorPipeline`` consuming a ``{"action": leader_dict}``
        transition and returning a ``np.ndarray`` of joint targets in radians.
        When ``None``, the default pipeline (deg-to-rad + wrist-roll offset) is
        constructed via
        :func:`so101_nexus_core.processors.pipelines.make_default_leader_action_pipeline`.
    """
    import gymnasium as gym

    from so101_nexus_core.processors.pipelines import make_default_leader_action_pipeline
    from so101_nexus_core.teleop.session import _recording_env_kwargs

    pipeline = action_pipeline or make_default_leader_action_pipeline(
        joint_names=joint_names,
        wrist_roll_offset_deg=wrist_roll_offset_deg,
    )

    for i in range(countdown, 0, -1):
        state.countdown_value = i
        time.sleep(1.0)
    state.countdown_value = 0

    env = gym.make(
        env_id,
        render_mode="rgb_array",
        **_recording_env_kwargs(env_id, wrist_wh, overhead_wh),
    )
    state.error = None
    try:
        leader_action = leader.get_action()
        init_qpos = pipeline({"action": leader_action})
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
            action = pipeline({"action": leader_action})
            obs, _, terminated, truncated, _ = env.step(action)

            # The leader arm action IS the observation.state for the dataset:
            # it matches what real robot joint encoders would report.
            state.episode_actions.append(action.astype(np.float32))
            state.episode_states.append(action.astype(np.float32))
            _publish_camera_frames(state, obs)

            if terminated or truncated:
                break

            sleep_time = frame_duration - (time.monotonic() - step_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

        state.episode_duration = time.monotonic() - start_time
    except Exception as exc:
        state.error = f"{type(exc).__name__}: {exc}"
    finally:
        env.close()
        state.is_recording = False
        state.should_stop = False
        state.recording_finished = True
