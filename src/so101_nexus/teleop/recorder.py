"""Shared recording-thread state and pure helpers for teleop.

Heavy dependencies (``gymnasium``, ``cv2``) are imported lazily inside
:func:`recording_thread` so that this module can be imported and unit-tested
on a base install without the ``teleop`` extra.
"""

from __future__ import annotations

import contextlib
import io
import logging
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np

if TYPE_CHECKING:
    from so101_nexus.teleop.config_customization import ConfigFactory, TeleopConfigOverrides
    from so101_nexus.teleop.leader import LeaderProtocol


PREVIEW_MAX_DIM = 320
logger = logging.getLogger(__name__)


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


class _InitialLeaderFollower(Protocol):
    def set_initial_leader_action(self, action: dict[str, Any]) -> None: ...


class _StepInfoLike(Protocol):
    terminated: bool
    reward: float


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
    episode_rewards: list[float] = field(default_factory=list)
    episode_wrist_images: list[np.ndarray] = field(default_factory=list)
    episode_overhead_images: list[np.ndarray] = field(default_factory=list)
    task_description: str = ""
    episode_duration: float = 0.0
    live_frame: np.ndarray | None = None
    live_overhead_frame: np.ndarray | None = None
    live_preview: np.ndarray | None = None
    terminated_at_frame: int | None = None

    episodes_completed: int = 0
    num_episodes: int = 0

    def clear_episode(self) -> None:
        """Reset all per-episode buffers."""
        self.episode_actions.clear()
        self.episode_states.clear()
        self.episode_rewards.clear()
        self.episode_wrist_images.clear()
        self.episode_overhead_images.clear()
        self.task_description = ""
        self.episode_duration = 0.0
        self.live_frame = None
        self.live_overhead_frame = None
        self.live_preview = None
        self.terminated_at_frame = None
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
        wrist_image = camera_obs.get("wrist")
        if wrist_image is None:
            wrist_image = camera_obs.get("wrist_camera")
        overhead_image = camera_obs.get("overhead")
        if overhead_image is None:
            overhead_image = camera_obs.get("overhead_camera")

    if wrist_image is not None:
        state.episode_wrist_images.append(wrist_image)
        state.live_frame = wrist_image
    if overhead_image is not None:
        state.episode_overhead_images.append(overhead_image)
        state.live_overhead_frame = overhead_image
    state.live_preview = _make_preview_frame(wrist_image, overhead_image)


def _seed_follower_from_leader(
    follower: _InitialLeaderFollower,
    leader: LeaderProtocol,
    wrist_roll_offset_deg: float,
) -> None:
    """Seed follower reset from the current leader pose when available."""
    from so101_nexus.teleop.leader import apply_wrist_roll_offset_deg

    try:
        initial_leader_action = leader.get_action()
    except Exception as exc:
        logger.warning("Failed to read initial leader pose: %s", exc)
        return
    follower.set_initial_leader_action(
        apply_wrist_roll_offset_deg(initial_leader_action, wrist_roll_offset_deg)
    )


def _should_stop_after_termination(
    state: RecordingState,
    step_info: _StepInfoLike | None,
    *,
    fps: int,
    success_hold_seconds: float,
) -> bool:
    """Update termination state and return whether the hold window elapsed."""
    if step_info is None or not step_info.terminated:
        return False
    if state.terminated_at_frame is None:
        state.terminated_at_frame = len(state.episode_actions)

    hold_frames = max(0, round(success_hold_seconds * fps))
    frames_since_success = len(state.episode_actions) - state.terminated_at_frame
    return frames_since_success >= hold_frames


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
    follower_calibration_dir,
    follower_robot_id: str,
    customization_overrides: TeleopConfigOverrides | None = None,
    env_config_profile: str | None = None,
    env_config_factory: ConfigFactory | None = None,
    success_hold_seconds: float = 0.5,
) -> None:
    """Run a countdown, then record by driving a ``SimSOFollower``."""
    from so101_nexus.lerobot_adapter.sim_follower import SimSOFollower
    from so101_nexus.teleop.leader import apply_wrist_roll_offset_deg
    from so101_nexus.teleop.session import (
        build_sim_follower_config,
        prepare_follower_calibration,
    )

    for i in range(countdown, 0, -1):
        state.countdown_value = i
        time.sleep(1.0)
    state.countdown_value = 0

    follower = None
    state.error = None
    try:
        prepare_follower_calibration(
            calibration_dir=follower_calibration_dir,
            robot_id=follower_robot_id,
        )
        follower_config = build_sim_follower_config(
            env_id=env_id,
            robot_id=follower_robot_id,
            wrist_wh=wrist_wh,
            overhead_wh=overhead_wh,
            fps=fps,
            calibration_dir=follower_calibration_dir,
            overrides=customization_overrides,
            profile_path=env_config_profile,
            factory=env_config_factory,
        )
        follower = SimSOFollower(follower_config)
        _seed_follower_from_leader(follower, leader, wrist_roll_offset_deg)
        follower.connect()

        state.clear_episode()
        env = follower._env
        if env is None:
            raise RuntimeError("follower environment is not connected after connect()")
        state.task_description = getattr(env.unwrapped, "task_description", "")
        state.is_recording = True

        frame_duration = 1.0 / fps
        start_time = time.monotonic()

        while not state.should_stop:
            step_start = time.monotonic()
            if len(state.episode_actions) >= max_steps:
                break

            leader_action = leader.get_action()
            action_dict = apply_wrist_roll_offset_deg(leader_action, wrist_roll_offset_deg)
            sent_action = follower.send_action(action_dict)
            step_info = follower.last_step_info()
            obs = follower.get_observation()

            _append_step_buffers(state, sent_action, obs, step_info, joint_names)

            if _should_stop_after_termination(
                state,
                step_info,
                fps=fps,
                success_hold_seconds=success_hold_seconds,
            ):
                break

            sleep_time = frame_duration - (time.monotonic() - step_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

        state.episode_duration = time.monotonic() - start_time
    except Exception as exc:
        state.error = f"{type(exc).__name__}: {exc}"
    finally:
        if follower is not None:
            with contextlib.suppress(Exception):
                follower.disconnect()
        state.is_recording = False
        state.should_stop = False
        state.recording_finished = True


def _dict_to_vector(
    motor_dict: Mapping[str, object],
    joint_names: tuple[str, ...],
) -> np.ndarray:
    """Extract ``<joint>.pos`` values in canonical joint order as float32."""
    return np.array(
        [float(cast("float", motor_dict[f"{name}.pos"])) for name in joint_names],
        dtype=np.float32,
    )


def _append_step_buffers(
    state: RecordingState,
    sent_action: Mapping[str, object],
    obs: Mapping[str, object],
    step_info: _StepInfoLike | None,
    joint_names: tuple[str, ...],
) -> None:
    """Append one step's actions, states, reward, and camera frames to buffers."""
    state.episode_actions.append(_dict_to_vector(sent_action, joint_names))
    state.episode_states.append(_dict_to_vector(obs, joint_names))
    # Reward from `env.step(a_t)` (captured by `SimSOFollower.send_action`)
    # aligns with this frame's action; default to 0.0 before the first step.
    state.episode_rewards.append(step_info.reward if step_info is not None else 0.0)
    _publish_camera_frames(state, obs)
