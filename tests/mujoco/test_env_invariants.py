"""Property-based invariants for every MuJoCo SO101-Nexus environment."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from so101_nexus.testing.invariants import (
    assert_env_seeded_reset_is_deterministic,
    assert_obs_always_in_observation_space,
    assert_random_actions_never_crash,
    assert_seeded_reset_is_deterministic,
)

ENV_IDS = [
    "MuJoCoTouch-v1",
    "MuJoCoLookAt-v1",
    "MuJoCoMove-v1",
    "MuJoCoPickLift-v1",
    "MuJoCoPickAndPlace-v1",
]


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_obs_always_in_observation_space(env_id, env_factory):
    """Observation returned by reset/step always belongs to ``observation_space``."""
    env = env_factory(task=env_id.removeprefix("MuJoCo").removesuffix("-v1"))

    @given(seed=st.integers(min_value=0, max_value=2**31 - 1))
    @settings(max_examples=20, deadline=None)
    def check(seed):
        # Each example resets all episode state while reusing the compiled scene.
        obs, _ = env.reset(seed=seed)
        env.action_space.seed(seed)
        assert env.observation_space.contains(obs)
        obs, reward, _, _, _ = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        assert np.isfinite(float(reward))

    check()


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_seeded_reset_is_deterministic(env_id, env_factory):
    env = env_factory(task=env_id.removeprefix("MuJoCo").removesuffix("-v1"))

    @given(seed=st.integers(min_value=0, max_value=2**31 - 1))
    @settings(max_examples=20, deadline=None)
    def check(seed):
        assert_env_seeded_reset_is_deterministic(env, seed)

    check()


@pytest.mark.parametrize(
    "check", [assert_obs_always_in_observation_space, assert_seeded_reset_is_deterministic]
)
def test_invariant_helpers_construct_registered_envs(check):
    import so101_nexus.mujoco  # noqa: F401

    check("MuJoCoTouch-v1", seed=0)


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_random_actions_never_crash(env_id):
    import so101_nexus.mujoco  # noqa: F401

    assert_random_actions_never_crash(env_id, steps=20)
