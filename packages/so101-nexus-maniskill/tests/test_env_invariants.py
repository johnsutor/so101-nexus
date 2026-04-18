"""Property-based invariants for every ManiSkill SO101-Nexus environment."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import so101_nexus_maniskill  # noqa: F401

BASE_KWARGS = {"obs_mode": "state", "num_envs": 1, "render_mode": None}

ENV_IDS = [
    "ManiSkillReachSO100-v1",
    "ManiSkillReachSO101-v1",
    "ManiSkillLookAtSO100-v1",
    "ManiSkillLookAtSO101-v1",
    "ManiSkillMoveSO100-v1",
    "ManiSkillMoveSO101-v1",
    "ManiSkillPickLiftSO100-v1",
    "ManiSkillPickLiftSO101-v1",
    "ManiSkillPickAndPlaceSO100-v1",
    "ManiSkillPickAndPlaceSO101-v1",
]


@pytest.mark.parametrize("env_id", ENV_IDS)
@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
@settings(
    max_examples=10,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_reward_is_finite(env_id, seed):
    env = gym.make(env_id, **BASE_KWARGS)
    try:
        env.reset(seed=seed)
        _, reward, _, _, _ = env.step(env.action_space.sample())
        assert np.isfinite(float(reward))
    finally:
        env.close()


@pytest.mark.parametrize("env_id", ENV_IDS)
@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
@settings(
    max_examples=10,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_seeded_reset_is_deterministic(env_id, seed):
    env = gym.make(env_id, **BASE_KWARGS)
    try:
        obs1, _ = env.reset(seed=seed)
        obs2, _ = env.reset(seed=seed)
        if isinstance(obs1, dict):
            assert obs1.keys() == obs2.keys()
            for k in obs1:
                np.testing.assert_array_equal(np.asarray(obs1[k]), np.asarray(obs2[k]))
        else:
            np.testing.assert_array_equal(np.asarray(obs1), np.asarray(obs2))
    finally:
        env.close()
