"""Dataset frame tests for policy rollout recording."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from so101_nexus.teleop.dataset import (
    OVERHEAD_KEY,
    REWARD_COMPONENT_FEATURE_KEYS,
    SIDE_KEY,
    WRIST_KEY,
)


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
        *REWARD_COMPONENT_FEATURE_KEYS,
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


class _RewardComponentsEnv(_DatasetEnv):
    """Env whose step info carries a reward-component breakdown."""

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        self.step_count += 1
        terminated = self.step_count >= self.terminate_after
        info = {
            "success": terminated,
            "reward_components": {"reaching": 0.4, "grasping": 0.1},
        }
        return self._obs(self.step_count), 0.5, terminated, False, info


def test_record_episode_writes_reward_components_from_info() -> None:
    from so101_nexus.policy_adapters import RolloutRecorder

    dataset = _Dataset()
    recorder = RolloutRecorder(
        _RewardComponentsEnv(terminate_after=1),
        _Policy(np.zeros(6, dtype=np.float32)),
        dataset=dataset,
    )

    recorder.record_episode()

    frame = dataset.frames[0]
    np.testing.assert_allclose(frame["reward_components.reaching"], [0.4])
    np.testing.assert_allclose(frame["reward_components.grasping"], [0.1])
    np.testing.assert_array_equal(frame["reward_components.task_objective"], [0.0])


class _SideVideoEnv(_DatasetEnv):
    """Env exposing the rgb_array render path used for the side video channel."""

    render_mode = "rgb_array"

    def __init__(self, *, terminate_after: int = 2) -> None:
        super().__init__(terminate_after=terminate_after)
        self.render_calls = 0

    def render(self) -> np.ndarray:
        self.render_calls += 1
        return np.full((4, 5, 3), 40 + self.step_count, dtype=np.uint8)


def test_record_side_video_writes_side_frames() -> None:
    from so101_nexus.policy_adapters import RolloutRecorder

    dataset = _Dataset()
    env = _SideVideoEnv(terminate_after=2)
    recorder = RolloutRecorder(
        env,
        _Policy(np.zeros(6, dtype=np.float32)),
        dataset=dataset,
        record_side_video=True,
    )

    recorder.record_episode()

    assert env.render_calls == 2
    for step, frame in enumerate(dataset.frames):
        # The side frame is rendered before env.step, so it depicts the same
        # pre-step state s_t as the rest of the frame.
        np.testing.assert_array_equal(frame[SIDE_KEY], np.full((4, 5, 3), 40 + step, np.uint8))


def test_record_side_video_keeps_side_out_of_policy_batches() -> None:
    from so101_nexus.policy_adapters import RolloutRecorder

    class _BatchSpy(_Policy):
        def __init__(self, action_deg: np.ndarray) -> None:
            super().__init__(action_deg)
            self.batches: list[dict[str, Any]] = []

        def select_action(self, batch: dict[str, Any]) -> np.ndarray:
            self.batches.append(batch)
            return super().select_action(batch)

    policy = _BatchSpy(np.zeros(6, dtype=np.float32))
    recorder = RolloutRecorder(
        _SideVideoEnv(terminate_after=1),
        policy,
        dataset=_Dataset(),
        record_side_video=True,
    )

    recorder.record_episode()

    image_keys = {k for k in policy.batches[0] if k.startswith("observation.images.")}
    assert image_keys == {"observation.images.overhead", "observation.images.wrist"}


def test_record_side_video_requires_rgb_array_render_mode() -> None:
    from so101_nexus.policy_adapters import RolloutRecorder

    with pytest.raises(ValueError, match="render_mode"):
        RolloutRecorder(
            _DatasetEnv(),
            _Policy(np.zeros(6, dtype=np.float32)),
            dataset=_Dataset(),
            record_side_video=True,
        )


def test_side_video_off_by_default() -> None:
    from so101_nexus.policy_adapters import RolloutRecorder

    dataset = _Dataset()
    env = _SideVideoEnv(terminate_after=1)
    recorder = RolloutRecorder(env, _Policy(np.zeros(6, dtype=np.float32)), dataset=dataset)

    recorder.record_episode()

    assert env.render_calls == 0
    assert SIDE_KEY not in dataset.frames[0]
