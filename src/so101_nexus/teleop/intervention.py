"""Reusable human-gated policy collection, independent of the viewer and backend."""

from __future__ import annotations

import queue
import threading
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any, Literal, Protocol

import numpy as np

from so101_nexus.config import SO101_JOINT_NAMES
from so101_nexus.teleop.leader import apply_wrist_roll_offset_deg

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from so101_nexus.policy_adapters.chunked_policy import ChunkedActionPolicy

ControlSource = Literal["policy", "human", "hold"]


class LeaderSynchronizer(Protocol):
    """Align and hold a leader until the operator explicitly releases it."""

    def start(self, target: dict[str, float]) -> None:
        """Start alignment to follower joint targets in LeRobot units."""

    def tick(self) -> bool:
        """Advance alignment and report completion with torque still enabled."""

    def cancel(self) -> None:
        """Cancel alignment and hold the measured leader pose."""

    @property
    def holding(self) -> bool:
        """Whether the last pose-hold command completed successfully."""
        ...

    def release(self) -> None:
        """Release torque after the operator supports the leader."""


class _InferenceWorker:
    def __init__(self, policy: ChunkedActionPolicy) -> None:
        self.policy = policy
        self.jobs: queue.Queue[tuple[Future, dict[str, Any], bool] | None] = queue.Queue()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def submit(self, batch: dict[str, Any], reset: bool) -> Future:
        future = Future()
        self.jobs.put((future, batch, reset))
        return future

    def _run(self) -> None:
        while (job := self.jobs.get()) is not None:
            future, batch, reset = job
            try:
                if reset:
                    self.policy.reset()
                future.set_result(self.policy.select_action(batch))
            except Exception as exc:
                future.set_exception(exc)

    def close(self) -> None:
        # Arbitrary model calls cannot be canceled; late results never control the robot.
        self.jobs.put(None)


class InterventionSession:
    """Arbitrate model and human actions with interruptible, serialized inference.

    Parameters
    ----------
    policy
        Model adapter consuming unbatched LeRobot observations: NumPy state,
        HWC uint8 images, and task text. Returns one absolute joint target in
        joint order, body degrees and gripper percent (0 to 100).
    synchronizer
        Aligns the leader with follower readback during every takeover.
    joint_names
        Joint order shared by observations, policy output, and follower commands.
    action_space
        Whether policy outputs absolute targets or deltas from current readback.
    """

    def __init__(
        self,
        policy: ChunkedActionPolicy,
        synchronizer: LeaderSynchronizer,
        joint_names: tuple[str, ...] = SO101_JOINT_NAMES,
        *,
        action_space: Literal["joint_pos", "joint_pos_delta"] = "joint_pos",
    ) -> None:
        if action_space not in ("joint_pos", "joint_pos_delta"):
            raise ValueError("policy action_space must be joint_pos or joint_pos_delta")
        self.action_space = action_space
        self.joint_names = joint_names
        self.synchronizer = synchronizer
        self._worker = _InferenceWorker(policy)
        self._lock = threading.Lock()
        self._requested: ControlSource = "hold"
        self._pause_after_sync = False
        self._version = 0
        self._applied_version = -1
        self._action_version = -1
        self._pending: tuple[int, Future] | None = None
        self._reset_policy = True
        self._interventions = 0
        self._closed = False
        self._parked_released = False
        self._leader_managed = False
        self._anchor_next_action = False
        self._human_offset: dict[str, float] | None = None
        self.status = "hold"
        self.synchronization_error: str | None = None
        self.source: ControlSource = "hold"

    @property
    def intervention_id(self) -> int:
        """Current correction segment, or zero for autonomous frames."""
        return self._interventions if self.source == "human" else 0

    def begin_episode(self) -> None:
        """Start autonomous collection and invalidate previous episode predictions."""
        with self._lock:
            if self._closed:
                raise RuntimeError("intervention session is closed")
            self._version += 1
            self._requested = "policy"
            self._pause_after_sync = False
            self._parked_released = False
        self._interventions = 0
        self._reset_policy = True
        self._human_offset = None
        self._anchor_next_action = False

    def request(self, source: ControlSource, *, pause_after_sync: bool = False) -> None:
        """Request a handoff, optionally waiting after leader alignment.

        With ``pause_after_sync=True``, human takeover stays paused until a
        subsequent ``request("human")`` starts correction without realigning.
        """
        if source not in ("policy", "human", "hold"):
            raise ValueError(f"Unknown control source: {source!r}")
        with self._lock:
            if self._closed:
                raise RuntimeError("intervention session is closed")
            pause_after_sync = source == "human" and pause_after_sync
            if (source, pause_after_sync) != (self._requested, self._pause_after_sync) or (
                source == "human" and self.status == "alignment paused"
            ):
                self._requested = source
                self._pause_after_sync = pause_after_sync
                self._version += 1

    def start_correction(self, *, support_token: tuple[int, int] | None = None) -> bool:
        """Start anchored correction from a held pose, rejecting stale commands."""
        with self._lock:
            if (
                self._closed
                or self._requested != "human"
                or not self._pause_after_sync
                or self._applied_version != self._version
                or self.status not in ("human paused", "alignment paused")
                or not self.synchronizer.holding
                or (support_token is not None and support_token != (id(self), self._version))
            ):
                return False
            self._pause_after_sync = False
            self._anchor_next_action = True
            self._version += 1
            return True

    def action(
        self,
        obs: Mapping[str, Any],
        task: str,
        human_action: Callable[[], dict[str, float]],
    ) -> dict[str, float] | None:
        """Return the next action, or hold simulation while waiting for a handoff."""
        with self._lock:
            version, requested = self._version, self._requested
            pause_after_sync = self._pause_after_sync
            if self._closed:
                return None
        target = {f"{name}.pos": float(obs[f"{name}.pos"]) for name in self.joint_names}
        if version != self._applied_version:
            self._apply_request(target, version, requested, pause_after_sync)
        self._action_version = version
        if self.status == "synchronizing":
            self._synchronize()
        if self.status == "human paused" and not pause_after_sync:
            return self._begin_correction(obs, human_action, version)
        if self.status in ("synchronizing", "alignment paused", "human paused"):
            return None
        if self.status == "human":
            return self._corrected_action(obs, human_action)
        if requested == "hold":
            return None
        self.source = "policy"
        return self._policy_action(obs, task, version)

    def _apply_request(
        self,
        target: dict[str, float],
        version: int,
        requested: ControlSource,
        pause_after_sync: bool,
    ) -> None:
        human_ready = (
            self.status in ("human paused", "alignment paused")
            and requested == "human"
            and not pause_after_sync
        )
        if human_ready and self.status == "alignment paused":
            self._interventions += 1
        if not human_ready:
            self._human_offset = None
            self._anchor_next_action = False
        self._leader_managed = True
        self.synchronization_error = None
        self.synchronizer.cancel()
        self._applied_version = version
        self.source = "hold"
        self.status = "human paused" if human_ready else requested
        if requested == "human" and not human_ready:
            self.status = "synchronizing"
            self.synchronizer.start(target)
        if requested == "policy":
            self._reset_policy = True

    def _begin_correction(
        self, obs: Mapping[str, Any], human_action: Callable[[], dict[str, float]], version: int
    ) -> dict[str, float] | None:
        with self._lock:
            if self._closed or version != self._version:
                return None
        action = self._corrected_action(obs, human_action)
        if not self._release_for_correction(version):
            return None
        self.status = "human"
        self.source = "human"
        return action

    def _corrected_action(
        self, obs: Mapping[str, Any], human_action: Callable[[], dict[str, float]]
    ) -> dict[str, float]:
        action = human_action()
        keys = [f"{name}.pos" for name in self.joint_names]
        if any(key not in action or not np.isfinite(action[key]) for key in keys):
            raise ValueError("Leader correction requires complete, finite joint readings")
        if self._anchor_next_action:
            target = {key: float(obs[key]) for key in keys}
            if not all(np.isfinite(value) for value in target.values()):
                raise ValueError("Follower correction anchor requires finite joint readings")
            self._human_offset = {key: target[key] - float(action[key]) for key in keys}
            self._anchor_next_action = False
            return target
        if self._human_offset is None:
            return action
        corrected = {key: float(action[key]) + self._human_offset[key] for key in keys}
        return apply_wrist_roll_offset_deg(
            corrected,
            0.0,
            reference_wrist_roll_deg=float(obs["wrist_roll.pos"])
            if "wrist_roll.pos" in obs
            else None,
        )

    def _synchronize(self) -> None:
        try:
            ready = self.synchronizer.tick()
        except TimeoutError as exc:
            self.synchronizer.cancel()
            self.synchronization_error = str(exc)
            self.status = "alignment paused"
            return
        if ready:
            self.status = "human paused"
            self._interventions += 1

    def _release_for_correction(self, version: int) -> bool:
        with self._lock:
            if self._closed or version != self._version:
                return False
            self.synchronizer.release()
            return True

    def _policy_action(
        self, obs: Mapping[str, Any], task: str, version: int
    ) -> dict[str, float] | None:
        if self._pending is not None:
            pending_version, future = self._pending
            if not future.done():
                self.status = "waiting for policy"
                return None
            self._pending = None
            if pending_version == version:
                output = future.result()
                if hasattr(output, "detach"):
                    output = output.detach().float().cpu().numpy()
                action = np.asarray(output, dtype=np.float32)
                if action.shape != (len(self.joint_names),) or not np.isfinite(action).all():
                    raise ValueError(
                        f"Policy must return a finite vector of {len(self.joint_names)} "
                        "joint targets."
                    )
                if self.action_space == "joint_pos_delta":
                    action = action + np.array([obs[f"{name}.pos"] for name in self.joint_names])
                self.status = "policy"
                return dict(
                    zip((f"{name}.pos" for name in self.joint_names), action.tolist(), strict=True)
                )
        batch: dict[str, Any] = {
            "observation.state": np.array(
                [obs[f"{name}.pos"] for name in self.joint_names], dtype=np.float32
            ),
            "task": task,
            **(
                {"observation.environment_state": np.array(obs["environment_state"], copy=True)}
                if "environment_state" in obs
                else {}
            ),
            **{
                f"observation.images.{key}": np.array(obs[key], copy=True)
                for key in ("wrist", "overhead", "side")
                if key in obs
            },
        }
        self._pending = (version, self._worker.submit(batch, self._reset_policy))
        self._reset_policy = False
        self.status = "waiting for policy"
        return None

    def execute(self, action: dict[str, float], send: Callable) -> Any | None:
        """Send an action only if no handoff arrived since it was selected."""
        with self._lock:
            if self._closed or self._action_version != self._version:
                return None
            return send(action)

    @property
    def support_token(self) -> tuple[int, int]:
        """Generation used to expire operator support after a control transition."""
        with self._lock:
            return id(self), self._version

    @property
    def leader_released(self) -> bool:
        """Whether the operator explicitly released the parked leader."""
        return self._parked_released

    def end_episode(self) -> None:
        """Invalidate outstanding inference and hold the physical leader pose."""
        self.request("hold")
        self.source = "hold"
        self.status = "hold"
        self._human_offset = None
        self._anchor_next_action = False
        if not self._parked_released:
            self._leader_managed = True
            self.synchronizer.cancel()

    def start_parking(self, target: dict[str, float]) -> None:
        """Begin a gradual return to follower-coordinate joint targets.

        Advance with ``synchronizer.tick()``. Call ``end_episode()`` on
        completion or cancellation to hold the measured pose.
        """
        with self._lock:
            if self._closed:
                raise RuntimeError("intervention session is closed")
            self._version += 1
            self._requested = "hold"
            self._parked_released = False
            self._leader_managed = True
            self.source = "hold"
            self.status = "returning to rest"
        self.synchronizer.start(target)

    def release_leader(self, *, support_token: tuple[int, int] | None = None) -> bool:
        """Explicitly release an idle leader, preserving release through shutdown."""
        with self._lock:
            if (
                self._closed
                or self._requested != "hold"
                or self.status != "hold"
                or (support_token is not None and support_token != (id(self), self._version))
            ):
                return False
            self.synchronizer.release()
            self._parked_released = True
            return True

    def close(self, *, wait: bool = False) -> None:
        """Hold the leader and reject requests, optionally waiting for inference.

        Set ``wait=True`` before reusing the policy in another session.
        """
        with self._lock:
            already_closed = self._closed
            self._closed = True
            self._version += 1
        try:
            if not already_closed and self._leader_managed and not self._parked_released:
                self.synchronizer.cancel()
        finally:
            if not already_closed:
                self._worker.close()
            if wait:
                self._worker.thread.join()


def intervention_chunk_mask(intervention_ids: Any, horizon: int) -> Any:
    """Mask targets outside a human segment, including padded chunk tails.

    Accepts one episode as a one-dimensional NumPy array or torch tensor. Positive IDs identify
    distinct human segments; zero identifies autonomous frames. Returns a boolean
    array or tensor of shape ``(frames, horizon)`` on the input tensor's device.
    """
    if horizon < 1 or intervention_ids.ndim != 1:
        raise ValueError("horizon must be positive and intervention_ids must be one-dimensional")
    if isinstance(intervention_ids, np.ndarray):
        offsets = np.arange(horizon)
        starts = np.arange(len(intervention_ids))[:, None]
        indices = starts + offsets
        padded = np.concatenate([intervention_ids, np.zeros(horizon, dtype=intervention_ids.dtype)])
        return (intervention_ids[:, None] > 0) & (padded[indices] == intervention_ids[:, None])
    import torch

    starts = torch.arange(len(intervention_ids), device=intervention_ids.device)[:, None]
    offsets = torch.arange(horizon, device=intervention_ids.device)
    padded = torch.cat([intervention_ids, intervention_ids.new_zeros(horizon)])
    return (intervention_ids[:, None] > 0) & (padded[starts + offsets] == intervention_ids[:, None])
