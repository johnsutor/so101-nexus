"""Dataset frame tests for policy rollout recording."""

from __future__ import annotations

from typing import Any

import numpy as np

from so101_nexus.teleop.dataset import OVERHEAD_KEY, WRIST_KEY


class _Box:
    def __init__(self) -> None:
        self.low = np.full(6, -10.0, dtype=np.float32)
        self.high = np.full(6, 10.0, dtype=np.float32)


class _DatasetEnv:
    def __init__(self, *, terminate_after: int = 2) -> None:
        self.action_space = _Box()
        self.task_description = "lift the cube"
        self.terminate_after = terminate_after
        self.step_count = 0

    @property
    def unwrapped(self) -> _DatasetEnv:
        return self

    def reset(self, *, seed: int | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        self.step_count = 0
        return self._obs(0), {}

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        self.step_count += 1
        terminated = self.step_count >= self.terminate_after
        return self._obs(self.step_count), 0.0, terminated, False, {"success": terminated}

    def _obs(self, step: int) -> dict[str, np.ndarray]:
        return {
            "state": np.array([0, np.pi / 2, -np.pi / 2, np.pi, step, -step], dtype=np.float32),
            "overhead_camera": np.full((2, 3, 3), step + 10, dtype=np.uint8),
            "wrist_camera": np.full((2, 3, 3), step + 20, dtype=np.uint8),
        }


class _Policy:
    def __init__(self, action_deg: np.ndarray) -> None:
        self.action_deg = action_deg

    def reset(self) -> None:
        pass

    def select_action(self, batch: dict[str, Any]) -> np.ndarray:
        return self.action_deg


class _Dataset:
    def __init__(self) -> None:
        self.frames: list[dict[str, Any]] = []
        self.save_episode_calls = 0

    def add_frame(self, frame: dict[str, Any]) -> None:
        self.frames.append(frame)

    def save_episode(self) -> None:
        self.save_episode_calls += 1


def test_record_episode_writes_lerobot_frames_in_degrees() -> None:
    from so101_nexus.policy_adapters import RolloutRecorder

    dataset = _Dataset()
    action_deg = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)
    recorder = RolloutRecorder(_DatasetEnv(), _Policy(action_deg), dataset=dataset)

    recorder.record_episode()

    assert dataset.save_episode_calls == 1
    assert len(dataset.frames) == 2
    first_frame = dataset.frames[0]
    assert set(first_frame) == {
        "observation.state",
        "action",
        "reward",
        "success",
        "done",
        "task",
        WRIST_KEY,
        OVERHEAD_KEY,
    }
    assert first_frame["task"] == "lift the cube"
    assert first_frame["reward"].shape == (1,)
    assert first_frame["reward"].dtype == np.float32
    np.testing.assert_allclose(first_frame["reward"], [0.0])
    np.testing.assert_array_equal(first_frame["action"], action_deg)
    np.testing.assert_allclose(
        first_frame["observation.state"],
        [0, 90, -90, 180, 0, 0],
        atol=1e-5,
    )
    assert first_frame[WRIST_KEY].dtype == np.uint8
    assert first_frame[OVERHEAD_KEY].dtype == np.uint8


def test_record_episode_omits_deselected_camera_frames() -> None:
    from so101_nexus.policy_adapters import RolloutRecorder

    dataset = _Dataset()
    recorder = RolloutRecorder(
        _DatasetEnv(terminate_after=1),
        _Policy(np.zeros(6, dtype=np.float32)),
        camera_keys=("wrist_camera",),
        dataset=dataset,
    )

    recorder.record_episode()

    assert WRIST_KEY in dataset.frames[0]
    assert OVERHEAD_KEY not in dataset.frames[0]
