"""Shared recording-thread state and pure helpers for teleop.

Heavy dependencies (``gymnasium``, ``cv2``) are imported lazily inside
:func:`recording_thread` so that this module can be imported and unit-tested
on a base install without the ``teleop`` extra.
"""

from __future__ import annotations

import contextlib
import io
import logging
import queue
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np

if TYPE_CHECKING:
    from so101_nexus.lerobot_adapter.sim_follower import SimSOFollower
    from so101_nexus.teleop.config_customization import ConfigFactory, TeleopConfigOverrides
    from so101_nexus.teleop.intervention import InterventionSession
    from so101_nexus.teleop.leader import LeaderProtocol


PREVIEW_MAX_DIM = 320
PREVIEW_INTERVAL_SECONDS = 0.1
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
    @property
    def terminated(self) -> bool: ...

    @property
    def truncated(self) -> bool: ...

    @property
    def reward(self) -> float: ...

    @property
    def info(self) -> dict[str, Any]: ...


@dataclass
class RecordingState:
    """Mutable state shared between the recording thread and the Gradio UI."""

    is_recording: bool = False
    should_stop: bool = False
    countdown_value: int = 0
    recording_finished: bool = False
    error: str | None = None

    episode_actions: list[np.ndarray] = field(default_factory=list)
    episode_intervention_ids: list[int] = field(default_factory=list)
    episode_states: list[np.ndarray] = field(default_factory=list)
    episode_rewards: list[float] = field(default_factory=list)
    episode_reward_components: list[dict[str, float]] = field(default_factory=list)
    episode_env_states: list[np.ndarray] = field(default_factory=list)
    episode_successes: list[float] = field(default_factory=list)
    episode_dones: list[float] = field(default_factory=list)
    episode_wrist_images: list[np.ndarray] = field(default_factory=list)
    episode_overhead_images: list[np.ndarray] = field(default_factory=list)
    episode_side_images: list[np.ndarray] = field(default_factory=list)
    task_description: str = ""
    placement_contract: dict[str, Any] | None = None
    episode_seed: int | None = None
    initial_sim_state: dict[str, Any] = field(default_factory=dict)
    rendering_metadata: dict[str, Any] = field(default_factory=dict)
    episode_duration: float = 0.0
    live_frame: np.ndarray | None = None
    live_overhead_frame: np.ndarray | None = None
    live_preview: np.ndarray | None = None
    _next_preview_at: float | None = field(default=None, init=False, repr=False)
    terminated_at_frame: int | None = None

    episodes_completed: int = 0
    num_episodes: int = 0

    def clear_episode(self) -> None:
        """Reset all per-episode buffers."""
        self.episode_actions.clear()
        self.episode_intervention_ids.clear()
        self.episode_states.clear()
        self.episode_rewards.clear()
        self.episode_reward_components.clear()
        self.episode_env_states.clear()
        self.episode_successes.clear()
        self.episode_dones.clear()
        self.episode_wrist_images.clear()
        self.episode_overhead_images.clear()
        self.episode_side_images.clear()
        self.task_description = ""
        self.placement_contract = None
        self.episode_seed = None
        self.initial_sim_state.clear()
        self.rendering_metadata.clear()
        self.episode_duration = 0.0
        self.live_frame = None
        self.live_overhead_frame = None
        self.live_preview = None
        self._next_preview_at = None
        self.terminated_at_frame = None
        self.recording_finished = False
        self.error = None

    def set_initial_provenance(self, seed: int, sim_state: dict[str, Any]) -> None:
        """Store the seed and post-reset simulator state for the current episode."""
        self.episode_seed = seed
        self.initial_sim_state = sim_state


def compute_delta_actions(actions: list[np.ndarray]) -> list[np.ndarray]:
    """Convert absolute joint positions to frame-to-frame deltas."""
    deltas: list[np.ndarray] = [np.zeros_like(actions[0])]
    for i in range(1, len(actions)):
        deltas.append(actions[i] - actions[i - 1])
    return deltas


def _make_preview_frame(
    wrist: np.ndarray | None,
    overhead: np.ndarray | None,
    side: np.ndarray | None = None,
    max_dim: int = PREVIEW_MAX_DIM,
) -> np.ndarray | None:
    """Combine available camera frames into one downscaled side-by-side preview."""
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
    for image in (wrist, overhead, side):
        if image is not None:
            tiles.append(_fit(image))
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


def _publish_camera_frames(state: RecordingState, obs: object, *, record: bool = True) -> None:
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
        if record:
            state.episode_wrist_images.append(wrist_image)
        state.live_frame = wrist_image
    if overhead_image is not None:
        if record:
            state.episode_overhead_images.append(overhead_image)
        state.live_overhead_frame = overhead_image
    side_image = obs.get("side") if isinstance(obs, Mapping) else None
    if record and side_image is not None:
        state.episode_side_images.append(side_image)
    if wrist_image is None and overhead_image is None and side_image is None:
        state.live_preview = None
        state._next_preview_at = None
        return
    now = time.monotonic()
    if state._next_preview_at is None or now >= state._next_preview_at:
        state.live_preview = _make_preview_frame(wrist_image, overhead_image, side_image)
        state._next_preview_at = now + PREVIEW_INTERVAL_SECONDS


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
    if step_info is None:
        return False
    if step_info.truncated:
        return True
    if state.terminated_at_frame is None:
        if not step_info.terminated:
            return False
        state.terminated_at_frame = len(state.episode_actions)

    hold_frames = max(0, round(success_hold_seconds * fps))
    frames_since_success = len(state.episode_actions) - state.terminated_at_frame
    return frames_since_success >= hold_frames


class _FollowerSession:
    def __init__(self) -> None:
        self.follower: SimSOFollower | None = None
        self.config_key: tuple[object, ...] | None = None

    def acquire(
        self,
        leader: LeaderProtocol,
        wrist_roll_offset_deg: float,
        episode_seed: int,
        **config_kwargs: Any,
    ) -> SimSOFollower:
        from so101_nexus.lerobot_adapter.sim_follower import SimSOFollower
        from so101_nexus.teleop.session import (
            build_sim_follower_config,
            prepare_follower_calibration,
        )

        config_key = tuple(config_kwargs.items())
        # Factories and profiles can change between episodes, so resolve them afresh.
        if (
            self.config_key != config_key
            or config_kwargs["profile_path"] is not None
            or config_kwargs["factory"] is not None
        ):
            self.close()
        if self.follower is None:
            prepare_follower_calibration(
                calibration_dir=config_kwargs["calibration_dir"],
                robot_id=config_kwargs["robot_id"],
            )
            config = build_sim_follower_config(**config_kwargs, seed=episode_seed)
            self.follower = SimSOFollower(config)
            _seed_follower_from_leader(self.follower, leader, wrist_roll_offset_deg)
            self.follower.connect()
            self.config_key = config_key
        else:
            _seed_follower_from_leader(self.follower, leader, wrist_roll_offset_deg)
            self.follower.reset(seed=episode_seed)
        return self.follower

    def close(self) -> None:
        if self.follower is not None:
            with contextlib.suppress(Exception):
                self.follower.disconnect()
            self.follower = None
        self.config_key = None


class RecordingWorker:
    """Keep simulator rendering and episode resets on one session-owned thread."""

    def __init__(self) -> None:
        self._requests: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._lock = threading.Lock()
        self._closed = False
        self._state: RecordingState | None = None
        self._done = threading.Event()
        self._done.set()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, **kwargs: Any) -> None:
        """Request an episode, rejecting overlapping recordings or a closed worker."""
        with self._lock:
            if self._closed:
                raise RuntimeError("recording worker is closed")
            if not self._done.is_set():
                raise RuntimeError("an episode is already recording")
            self._state = kwargs["state"]
            self._state.should_stop = False
            self._state.recording_finished = False
            self._done.clear()
            self._requests.put(kwargs)

    def wait(self, timeout: float | None = None) -> bool:
        """Wait until the current episode has completed."""
        return self._done.wait(timeout)

    def close(self) -> None:
        """Cancel recording and disconnect the simulator on its owning thread."""
        with self._lock:
            if not self._closed:
                self._closed = True
                if self._state is not None:
                    self._state.should_stop = True
                self._requests.put(None)
        self._thread.join()

    def _run(self) -> None:
        follower_session = _FollowerSession()
        try:
            while (kwargs := self._requests.get()) is not None:
                try:
                    recording_thread(**kwargs, _follower_session=follower_session)
                finally:
                    self._done.set()
        finally:
            follower_session.close()


def _run_countdown(state: RecordingState, countdown: int) -> bool:
    for value in range(countdown, 0, -1):
        state.countdown_value = value
        # Short waits keep session shutdown responsive during a countdown.
        for _ in range(10):
            if state.should_stop:
                return False
            time.sleep(0.1)
    state.countdown_value = 0
    return not state.should_stop


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
    seed: int = 0,
    _follower_session: _FollowerSession | None = None,
    intervention: InterventionSession | None = None,
) -> None:
    """Run a countdown, then record by driving a ``SimSOFollower``."""
    follower_session = _follower_session or _FollowerSession()
    state.clear_episode()
    try:
        if not _run_countdown(state, countdown):
            return
        episode_seed = seed + state.episodes_completed
        follower = follower_session.acquire(
            leader,
            wrist_roll_offset_deg,
            episode_seed,
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

        state.set_initial_provenance(episode_seed, follower.initial_state_provenance())
        env = follower._env
        if env is None:
            raise RuntimeError("follower environment is not connected after connect()")
        state.task_description = getattr(env.unwrapped, "task_description", "")
        if intervention is not None:
            intervention.begin_episode()
        state.is_recording = True

        frame_duration = 1.0 / fps
        start_time = time.monotonic()

        while not state.should_stop:
            step_start = time.monotonic()
            if len(state.episode_actions) >= max_steps:
                break
            if not _record_step(
                state, follower, leader, intervention, joint_names, wrist_roll_offset_deg
            ):
                time.sleep(frame_duration)
                continue
            step_info = follower.last_step_info()

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
        state.rendering_metadata = follower.rendering_metadata()
    except Exception as exc:
        state.error = f"{type(exc).__name__}: {exc}"
        follower_session.close()
    finally:
        _finish_recording(state, follower_session, _follower_session is None, intervention)


def _finish_recording(
    state: RecordingState,
    follower_session: _FollowerSession,
    close_follower: bool,
    intervention: InterventionSession | None,
) -> None:
    if intervention is not None:
        try:
            intervention.end_episode()
        except Exception as exc:
            state.error = state.error or f"Failed to release leader: {exc}"
    if close_follower:
        follower_session.close()
    state.countdown_value = 0
    state.is_recording = False
    state.should_stop = False
    state.recording_finished = True


def _record_step(
    state: RecordingState,
    follower: SimSOFollower,
    leader: LeaderProtocol,
    intervention: InterventionSession | None,
    joint_names: tuple[str, ...],
    wrist_roll_offset_deg: float,
) -> bool:
    from so101_nexus.teleop.leader import apply_wrist_roll_offset_deg

    def human_action() -> dict[str, float]:
        return apply_wrist_roll_offset_deg(leader.get_action(), wrist_roll_offset_deg)

    obs = follower.get_observation()
    if intervention is None:
        sent_action = follower.send_action(human_action())
    else:
        action = intervention.action(obs, state.task_description, human_action)
        sent_action = (
            intervention.execute(action, follower.send_action)
            if action is not None and not state.should_stop
            else None
        )
        if sent_action is None:
            _publish_camera_frames(state, obs, record=False)
            return False
        state.episode_intervention_ids.append(intervention.intervention_id)
    _append_step_buffers(state, sent_action, obs, follower.last_step_info(), joint_names)
    return True


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
    """Append one step's actions, states, reward, success, done, env state, and cameras."""
    state.episode_actions.append(_dict_to_vector(sent_action, joint_names))
    state.episode_states.append(_dict_to_vector(obs, joint_names))
    # Reward/success/done from `env.step(a_t)` (captured by
    # `SimSOFollower.send_action`) align with this frame's action; default before
    # the first step. `done` is `terminated or truncated`; `success` is the env's
    # `info["success"]` (0.0 when the env sets no success key). `reward_components`
    # is the per-facet breakdown the env stored on `info` (empty when absent).
    state.episode_rewards.append(step_info.reward if step_info is not None else 0.0)
    state.episode_reward_components.append(
        step_info.info.get("reward_components", {}) if step_info is not None else {}
    )
    if step_info is None:
        state.episode_successes.append(0.0)
        state.episode_dones.append(0.0)
    else:
        state.placement_contract = step_info.info.get("placement_contract")
        state.episode_successes.append(float(bool(step_info.info.get("success", False))))
        state.episode_dones.append(float(step_info.terminated or step_info.truncated))
    # Privileged state is the pre-step observation (s_t), aligned with
    # `observation.state`. Absent when the env exposes no non-camera components.
    env_state = obs.get("environment_state")
    if env_state is not None:
        state.episode_env_states.append(np.asarray(env_state, dtype=np.float32))
    _publish_camera_frames(state, obs)
