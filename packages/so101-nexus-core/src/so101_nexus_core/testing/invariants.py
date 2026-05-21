"""Shared backend invariant assertions used by package test suites."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any


def _make_env(env_id: str, base_kwargs: Mapping[str, Any] | None = None):
    """Create a Gymnasium env with optional backend-specific kwargs."""
    import gymnasium as gym

    return gym.make(env_id, **dict(base_kwargs or {}))


def _assert_observations_equal(obs1: object, obs2: object) -> None:
    """Assert two reset observations are identical, including dict observations."""
    if isinstance(obs1, dict):
        assert isinstance(obs2, dict)
        obs1_map = cast("dict[Any, Any]", obs1)
        obs2_map = cast("dict[Any, Any]", obs2)
        assert obs1_map.keys() == obs2_map.keys()
        for key in obs1_map:
            np.testing.assert_array_equal(np.asarray(obs1_map[key]), np.asarray(obs2_map[key]))
    else:
        np.testing.assert_array_equal(np.asarray(obs1), np.asarray(obs2))


def assert_obs_always_in_observation_space(
    env_id: str,
    seed: int,
    *,
    base_kwargs: Mapping[str, Any] | None = None,
) -> None:
    """Assert reset/step observations remain in the env observation space."""
    env = _make_env(env_id, base_kwargs)
    try:
        obs, _ = env.reset(seed=seed)
        assert env.observation_space.contains(obs)
        obs, reward, _, _, _ = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        assert np.isfinite(float(reward))
    finally:
        env.close()


def assert_reward_is_finite(
    env_id: str,
    seed: int,
    *,
    base_kwargs: Mapping[str, Any] | None = None,
) -> None:
    """Assert one sampled action produces a finite scalar reward."""
    env = _make_env(env_id, base_kwargs)
    try:
        assert_env_reward_is_finite(env, seed)
    finally:
        env.close()


def assert_env_reward_is_finite(env: Any, seed: int) -> None:
    """Assert one sampled action in an existing environment produces a finite reward."""
    env.reset(seed=seed)
    _, reward, _, _, _ = env.step(env.action_space.sample())
    assert np.isfinite(float(reward))


def assert_seeded_reset_is_deterministic(
    env_id: str,
    seed: int,
    *,
    base_kwargs: Mapping[str, Any] | None = None,
) -> None:
    """Assert two resets with the same seed return identical observations."""
    env = _make_env(env_id, base_kwargs)
    try:
        assert_env_seeded_reset_is_deterministic(env, seed)
    finally:
        env.close()


def assert_env_seeded_reset_is_deterministic(env: Any, seed: int) -> None:
    """Assert two same-seed resets in an existing environment return identical observations."""
    obs1, _ = env.reset(seed=seed)
    obs2, _ = env.reset(seed=seed)
    _assert_observations_equal(obs1, obs2)


def assert_random_actions_never_crash(
    env_id: str,
    *,
    steps: int,
    base_kwargs: Mapping[str, Any] | None = None,
) -> None:
    """Assert a short rollout of sampled actions completes without raising."""
    env = _make_env(env_id, base_kwargs)
    try:
        env.reset(seed=0)
        for _ in range(steps):
            env.step(env.action_space.sample())
    finally:
        env.close()
