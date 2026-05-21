"""Property-based invariants for every ManiSkill SO101-Nexus environment."""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import so101_nexus_maniskill  # noqa: F401
from so101_nexus_core.testing.invariants import (
    assert_env_reward_is_finite,
    assert_env_seeded_reset_is_deterministic,
)

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


@pytest.fixture
def maniskill_env(request):
    """Create one ManiSkill env per parametrized invariant case."""
    import gymnasium as gym

    env = gym.make(request.param, **BASE_KWARGS)
    try:
        yield env
    finally:
        env.close()


@pytest.mark.parametrize("maniskill_env", ENV_IDS, indirect=True, ids=ENV_IDS)
@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
@settings(
    max_examples=10,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_reward_is_finite(maniskill_env, seed):
    assert_env_reward_is_finite(maniskill_env, seed)


@pytest.mark.parametrize("maniskill_env", ENV_IDS, indirect=True, ids=ENV_IDS)
@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
@settings(
    max_examples=10,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_seeded_reset_is_deterministic(maniskill_env, seed):
    assert_env_seeded_reset_is_deterministic(maniskill_env, seed)
