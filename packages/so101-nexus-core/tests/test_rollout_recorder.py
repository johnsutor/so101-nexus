"""Tests for policy rollout recording against SO101-Nexus envs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest


@dataclass
class _Box:
    low: np.ndarray
    high: np.ndarray


class _RolloutEnv:
    def __init__(self, *, terminate_after: int = 2) -> None:
        self.action_space = _Box(
            low=np.array([-1, -1, -1, -1, -1, -0.2], dtype=np.float32),
            high=np.array([1, 1, 1, 1, 1, 1.2], dtype=np.float32),
        )
        self.task_description = "reach the cube"
        self.terminate_after = terminate_after
        self.reset_seeds: list[int | None] = []
        self.actions_rad: list[np.ndarray] = []
        self.step_count = 0

    @property
    def unwrapped(self) -> _RolloutEnv:
        return self

    def reset(self, *, seed: int | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        self.reset_seeds.append(seed)
        self.step_count = 0
        return self._obs(0), {"seed": seed}

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        self.actions_rad.append(np.asarray(action, dtype=np.float32))
        self.step_count += 1
        terminated = self.step_count >= self.terminate_after
        return self._obs(self.step_count), 0.0, terminated, False, {"success": terminated}

    def _obs(self, step: int) -> dict[str, np.ndarray]:
        return {
            "state": np.array(
                [0.0, np.pi / 2, -np.pi / 2, np.pi, float(step), -float(step)],
                dtype=np.float32,
            ),
            "overhead_camera": np.full((3, 4, 3), 10 + step, dtype=np.uint8),
            "wrist_camera": np.full((3, 4, 3), 20 + step, dtype=np.uint8),
        }


class _Policy:
    def __init__(self, actions_deg: list[np.ndarray]) -> None:
        self.actions_deg = actions_deg
        self.reset_calls = 0
        self.batches: list[dict[str, Any]] = []

    def reset(self) -> None:
        self.reset_calls += 1

    def select_action(self, batch: dict[str, Any]) -> np.ndarray:
        self.batches.append(batch)
        if not self.actions_deg:
            raise AssertionError("select_action called more times than expected")
        return self.actions_deg.pop(0)


def test_record_episode_converts_units_and_stops_on_termination() -> None:
    from so101_nexus_core.policy_adapters import RolloutRecorder

    first_action_deg = np.array([180, -180, 90, 0, 0, 100], dtype=np.float32)
    second_action_deg = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)
    env = _RolloutEnv()
    policy = _Policy([first_action_deg, second_action_deg])
    recorder = RolloutRecorder(env, policy, max_steps_per_episode=10)

    result = recorder.record_episode(seed=123)

    assert policy.reset_calls == 1
    assert result.success is True
    assert result.n_steps == 2
    assert env.reset_seeds == [123]
    expected_first_action = np.clip(
        np.deg2rad(first_action_deg),
        env.action_space.low,
        env.action_space.high,
    )
    np.testing.assert_allclose(env.actions_rad[0], expected_first_action)
    np.testing.assert_array_equal(result.actions_deg[0], first_action_deg)
    np.testing.assert_allclose(result.states_deg[0], [0, 90, -90, 180, 0, 0], atol=1e-5)

    first_batch = policy.batches[0]
    assert first_batch["task"] == "reach the cube"
    assert first_batch["observation.state"].dtype == np.float32
    np.testing.assert_allclose(
        first_batch["observation.state"],
        [0, 90, -90, 180, 0, 0],
        atol=1e-5,
    )
    assert first_batch["observation.images.overhead"].shape == (3, 4, 3)
    assert first_batch["observation.images.wrist"].shape == (3, 4, 3)


def test_record_episode_uses_explicit_task_over_env_task() -> None:
    from so101_nexus_core.policy_adapters import RolloutRecorder

    env = _RolloutEnv(terminate_after=1)
    policy = _Policy([np.zeros(6, dtype=np.float32)])
    recorder = RolloutRecorder(env, policy, task="explicit task")

    recorder.record_episode()

    assert policy.batches[0]["task"] == "explicit task"


def test_record_episodes_increments_seed() -> None:
    from so101_nexus_core.policy_adapters import RolloutRecorder

    env = _RolloutEnv(terminate_after=1)
    policy = _Policy([np.zeros(6, dtype=np.float32), np.zeros(6, dtype=np.float32)])
    recorder = RolloutRecorder(env, policy)

    results = recorder.record_episodes(2, seed=7)

    assert [result.n_steps for result in results] == [1, 1]
    assert env.reset_seeds == [7, 8]
    assert policy.reset_calls == 2


def test_record_episodes_preserves_none_seed() -> None:
    from so101_nexus_core.policy_adapters import RolloutRecorder

    env = _RolloutEnv(terminate_after=1)
    policy = _Policy([np.zeros(6, dtype=np.float32), np.zeros(6, dtype=np.float32)])
    recorder = RolloutRecorder(env, policy)

    recorder.record_episodes(2)

    assert env.reset_seeds == [None, None]


def test_invalid_camera_key_raises() -> None:
    from so101_nexus_core.policy_adapters import RolloutRecorder

    with pytest.raises(ValueError, match="camera_keys"):
        RolloutRecorder(
            _RolloutEnv(),
            _Policy([]),
            camera_keys=("side_camera",),
        )
