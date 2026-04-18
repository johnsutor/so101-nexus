"""Property-based invariants for every MuJoCo SO101-Nexus environment."""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import so101_nexus_mujoco  # noqa: F401

ENV_IDS = [
    "MuJoCoReach-v1",
    "MuJoCoLookAt-v1",
    "MuJoCoMove-v1",
    "MuJoCoPickLift-v1",
    "MuJoCoPickAndPlace-v1",
]


@pytest.mark.parametrize("env_id", ENV_IDS)
@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_obs_always_in_observation_space(env_id, seed):
    """Observation returned by reset/step always belongs to ``observation_space``."""
    env = gym.make(env_id)
    try:
        obs, _ = env.reset(seed=seed)
        assert env.observation_space.contains(obs)
        obs, reward, term, trunc, info = env.step(env.action_space.sample())
        assert np.isfinite(float(reward))
    finally:
        env.close()


@pytest.mark.parametrize("env_id", ENV_IDS)
@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_seeded_reset_is_deterministic(env_id, seed):
    env = gym.make(env_id)
    try:
        obs1, _ = env.reset(seed=seed)
        obs2, _ = env.reset(seed=seed)
        if isinstance(obs1, dict):
            assert obs1.keys() == obs2.keys()
            for k in obs1:
                np.testing.assert_array_equal(obs1[k], obs2[k])
        else:
            np.testing.assert_array_equal(obs1, obs2)
    finally:
        env.close()


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_random_actions_never_crash(env_id):
    env = gym.make(env_id)
    try:
        env.reset(seed=0)
        for _ in range(20):
            env.step(env.action_space.sample())
    finally:
        env.close()
