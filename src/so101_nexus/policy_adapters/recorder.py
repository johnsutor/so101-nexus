"""Rollout recording for chunked policies and SO101-Nexus environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from so101_nexus.config import SO101_JOINT_NAMES

if TYPE_CHECKING:
    from collections.abc import Sequence

    from so101_nexus.policy_adapters.chunked_policy import ChunkedActionPolicy

SUPPORTED_CAMERA_KEYS = frozenset({"overhead_camera", "wrist_camera"})


@dataclass
class EpisodeResult:
    """Summary data returned after one rollout episode."""

    success: bool
    n_steps: int
    actions_deg: np.ndarray
    states_deg: np.ndarray
    info: dict[str, Any]


class RolloutRecorder:
    """Run a chunked-action policy against a SO101-Nexus Gymnasium env.

    Parameters
    ----------
    env : Any
        Gymnasium-compatible environment exposing ``state`` in radians and
        camera observations in HWC ``uint8`` arrays.
    policy : ChunkedActionPolicy
        Policy that accepts LeRobot-shaped batches and returns one action.
    camera_keys : Sequence[str], optional
        Env camera keys to map into ``observation.images.<stem>`` batch keys.
        Supported keys are ``"overhead_camera"`` and ``"wrist_camera"``. Keep
        these keys and their order aligned with ``MolmoActPolicy.image_keys`` if
        either side is customized.
    joint_names : tuple[str, ...], optional
        Joint order expected by policy actions and env states.
    task : str, optional
        Task string written to the policy batch and dataset frames. Defaults to
        ``env.unwrapped.task_description`` when available.
    max_steps_per_episode : int, optional
        Maximum number of env steps to record per episode.
    dataset : Any, optional
        LeRobot-like dataset. When provided, frames are appended with
        ``teleop.dataset.build_frame`` and ``save_episode`` is called at the
        episode boundary.
    fps : int, optional
        Dataset frame rate metadata retained for callers constructing matching
        LeRobot datasets.

    Notes
    -----
    Environment runtime state and actions stay in radians. Policy batches,
    returned policy actions, and dataset frames use LeRobot degree units. The
    recorder treats policy actions as absolute joint positions, then converts to
    radians and clips to ``env.action_space`` before stepping.
    """

    def __init__(
        self,
        env: Any,
        policy: ChunkedActionPolicy,
        *,
        camera_keys: Sequence[str] = ("overhead_camera", "wrist_camera"),
        joint_names: tuple[str, ...] = SO101_JOINT_NAMES,
        task: str | None = None,
        max_steps_per_episode: int = 200,
        dataset: Any | None = None,
        fps: int = 30,
    ) -> None:
        unknown_camera_keys = set(camera_keys) - SUPPORTED_CAMERA_KEYS
        if unknown_camera_keys:
            raise ValueError(
                "camera_keys must contain only 'overhead_camera' and 'wrist_camera'; "
                f"got {sorted(unknown_camera_keys)}."
            )
        if max_steps_per_episode <= 0:
            raise ValueError("max_steps_per_episode must be positive.")
        if fps <= 0:
            raise ValueError("fps must be positive.")

        self.env = env
        self.policy = policy
        self.camera_keys = tuple(camera_keys)
        self.joint_names = joint_names
        self.task = task
        self.max_steps_per_episode = max_steps_per_episode
        self.dataset = dataset
        self.fps = fps

    def record_episode(self, *, seed: int | None = None) -> EpisodeResult:
        """Record one policy rollout episode."""
        obs, _ = self.env.reset(seed=seed)
        self.policy.reset()
        task = self.task
        if task is None:
            task = getattr(self.env.unwrapped, "task_description", "")

        actions_deg: list[np.ndarray] = []
        states_deg: list[np.ndarray] = []
        info: dict[str, Any] = {}

        for _ in range(self.max_steps_per_episode):
            batch = self._build_batch(obs, task)
            action_deg = np.asarray(self.policy.select_action(batch), dtype=np.float32)
            if action_deg.shape != (len(self.joint_names),):
                raise ValueError(
                    "Policy action must have shape "
                    f"({len(self.joint_names)},); got {action_deg.shape}."
                )

            if self.dataset is not None:
                self._add_frame(obs, action_deg, task)

            actions_deg.append(action_deg.astype(np.float32))
            states_deg.append(np.rad2deg(obs["state"]).astype(np.float32))

            action_rad = np.clip(
                np.deg2rad(action_deg),
                self.env.action_space.low,
                self.env.action_space.high,
            ).astype(np.float32)
            obs, _, terminated, truncated, info = self.env.step(action_rad)
            if terminated or truncated:
                break

        if self.dataset is not None:
            self.dataset.save_episode()

        return EpisodeResult(
            success=bool(info.get("success", False)),
            n_steps=len(actions_deg),
            actions_deg=np.asarray(actions_deg, dtype=np.float32),
            states_deg=np.asarray(states_deg, dtype=np.float32),
            info=info,
        )

    def record_episodes(self, n: int, *, seed: int | None = None) -> list[EpisodeResult]:
        """Record ``n`` episodes, incrementing integer seeds when provided."""
        if n < 0:
            raise ValueError("n must be non-negative.")
        return [
            self.record_episode(seed=None if seed is None else seed + episode_idx)
            for episode_idx in range(n)
        ]

    def _build_batch(self, obs: dict[str, np.ndarray], task: str) -> dict[str, Any]:
        """Build the LeRobot-shaped policy batch for one env observation."""
        images = {
            f"observation.images.{key.removesuffix('_camera')}": obs[key]
            for key in self.camera_keys
        }
        return {
            "observation.state": np.rad2deg(obs["state"]).astype(np.float32),
            "task": task,
            **images,
        }

    def _add_frame(self, obs: dict[str, np.ndarray], action_deg: np.ndarray, task: str) -> None:
        """Append one frame to the configured dataset."""
        from so101_nexus.teleop.dataset import FieldSelection, build_frame

        if self.dataset is None:
            raise RuntimeError("RolloutRecorder has no dataset configured.")

        selection = FieldSelection(
            wrist_image="wrist_camera" in self.camera_keys,
            overhead_image="overhead_camera" in self.camera_keys,
            task=True,
        )
        frame = build_frame(
            selection,
            state=np.rad2deg(obs["state"]).astype(np.float32),
            action=action_deg.astype(np.float32),
            task=task,
            wrist_image=obs.get("wrist_camera"),
            overhead_image=obs.get("overhead_camera"),
        )
        dataset = self.dataset
        dataset.add_frame(frame)
