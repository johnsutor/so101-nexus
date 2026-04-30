"""Tests for the LeRobotEnvWrapper."""

from __future__ import annotations

import numpy as np
import pytest

gymnasium = pytest.importorskip("gymnasium")
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402


class _FakeSO101Env(gym.Env):
    """Minimal stand-in for an so101-nexus env: state + two camera frames."""

    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32),
                "wrist_camera": gym.spaces.Box(low=0, high=255, shape=(8, 8, 3), dtype=np.uint8),
                "overhead_camera": gym.spaces.Box(low=0, high=255, shape=(8, 8, 3), dtype=np.uint8),
            }
        )
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):  # type: ignore[override]
        super().reset(seed=seed)
        return self._obs(), {}

    def step(self, action):  # type: ignore[override]
        return self._obs(), 0.0, False, False, {}

    def _obs(self) -> dict:
        return {
            "state": np.zeros(6, dtype=np.float32),
            "wrist_camera": np.full((8, 8, 3), 255, dtype=np.uint8),
            "overhead_camera": np.zeros((8, 8, 3), dtype=np.uint8),
        }


def test_wrapper_renames_keys_and_converts_images() -> None:
    from so101_nexus_core.processors.lerobot_env_wrapper import LeRobotEnvWrapper

    env = LeRobotEnvWrapper(_FakeSO101Env())
    obs, _ = env.reset()

    assert set(obs.keys()) == {
        "observation.state",
        "observation.images.wrist",
        "observation.images.overhead",
    }
    assert isinstance(obs["observation.images.wrist"], torch.Tensor)
    assert obs["observation.images.wrist"].shape == (3, 8, 8)
    assert obs["observation.images.wrist"].dtype == torch.float32
    assert torch.allclose(
        obs["observation.images.wrist"],
        torch.ones_like(obs["observation.images.wrist"]),
    )


def test_wrapper_step_passes_through_reward_done_truncated_info() -> None:
    from so101_nexus_core.processors.lerobot_env_wrapper import LeRobotEnvWrapper

    env = LeRobotEnvWrapper(_FakeSO101Env())
    env.reset()
    action = np.zeros(6, dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)

    assert reward == 0.0
    assert terminated is False
    assert truncated is False
    assert info == {}
    assert "observation.state" in obs


def test_wrapper_observation_space_uses_lerobot_keys() -> None:
    from so101_nexus_core.processors.lerobot_env_wrapper import LeRobotEnvWrapper

    env = LeRobotEnvWrapper(_FakeSO101Env())
    keys = set(env.observation_space.spaces.keys())

    assert "observation.state" in keys
    assert "observation.images.wrist" in keys
    assert "observation.images.overhead" in keys
    assert "state" not in keys
    assert "wrist_camera" not in keys


def test_wrapper_add_batch_dim_unsqueezes_images() -> None:
    from so101_nexus_core.processors.lerobot_env_wrapper import LeRobotEnvWrapper

    env = LeRobotEnvWrapper(_FakeSO101Env(), add_batch_dim=True)
    obs, _ = env.reset()

    assert obs["observation.images.wrist"].shape == (1, 3, 8, 8)


def test_wrapper_rejects_non_dict_observation_space() -> None:
    from so101_nexus_core.processors.lerobot_env_wrapper import LeRobotEnvWrapper

    class _BoxOnlyEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self) -> None:
            super().__init__()
            self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(6,))
            self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(6,))

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return np.zeros(6), {}

        def step(self, action):
            return np.zeros(6), 0.0, False, False, {}

    with pytest.raises(TypeError, match="Dict observation_space"):
        LeRobotEnvWrapper(_BoxOnlyEnv())


def test_make_lerobot_env_factory_returns_wrapped_env() -> None:
    """The factory wraps a registered env id."""
    from so101_nexus_core.processors import make_lerobot_env

    gym.register(id="LeRobotWrapperTest-v0", entry_point=_FakeSO101Env)
    try:
        env = make_lerobot_env("LeRobotWrapperTest-v0")
        obs, _ = env.reset()
        assert "observation.state" in obs
    finally:
        gym.envs.registration.registry.pop("LeRobotWrapperTest-v0", None)
