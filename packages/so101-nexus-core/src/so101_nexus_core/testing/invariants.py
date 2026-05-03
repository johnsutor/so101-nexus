"""Hypothesis-based shared env invariant test factories.

Returns three test functions that verify reset/step contracts across any
list of registered Gymnasium env ids. ManiSkill envs use tensor obs that
do not satisfy `observation_space.contains`, so callers can disable the
obs-in-space check with `check_obs_in_space=False`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import gymnasium as gym
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

if TYPE_CHECKING:
    from collections.abc import Callable


def register_env_invariant_tests(
    env_ids: list[str],
    *,
    base_kwargs: dict[str, Any] | None = None,
    max_examples: int = 10,
    check_obs_in_space: bool = True,
    db_namespace: str = "",
) -> tuple[Callable, Callable, Callable]:
    """Build three parametrized hypothesis tests for the given env ids.

    Parameters
    ----------
    db_namespace
        Suffix appended to each generated test's ``__qualname__`` so that
        Hypothesis keys its example database independently per caller.
        Pass a unique value per backend (e.g. ``"mujoco"``, ``"maniskill"``)
        when the same factory is invoked from multiple test modules.

    Returns
    -------
    (test_obs_in_space_or_finite_reward,
     test_seeded_reset_is_deterministic,
     test_random_actions_never_crash)
    """
    kwargs = dict(base_kwargs or {})
    suffix = f"__{db_namespace}" if db_namespace else ""

    @pytest.mark.parametrize("env_id", env_ids)
    @given(seed=st.integers(min_value=0, max_value=2**31 - 1))
    @settings(
        max_examples=max_examples,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_obs_or_reward(env_id, seed):
        env = gym.make(env_id, **kwargs)
        try:
            obs, _ = env.reset(seed=seed)
            if check_obs_in_space:
                assert env.observation_space.contains(obs)
            _, reward, _, _, _ = env.step(env.action_space.sample())
            assert np.isfinite(float(reward))
        finally:
            env.close()

    @pytest.mark.parametrize("env_id", env_ids)
    @given(seed=st.integers(min_value=0, max_value=2**31 - 1))
    @settings(
        max_examples=max_examples,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_seeded_reset_deterministic(env_id, seed):
        env = gym.make(env_id, **kwargs)
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

    @pytest.mark.parametrize("env_id", env_ids)
    def test_random_actions_safe(env_id):
        env = gym.make(env_id, **kwargs)
        try:
            env.reset(seed=0)
            for _ in range(20):
                env.step(env.action_space.sample())
        finally:
            env.close()

    test_obs_or_reward.__qualname__ = (
        f"register_env_invariant_tests{suffix}.<locals>.test_obs_or_reward"
    )
    test_seeded_reset_deterministic.__qualname__ = (
        f"register_env_invariant_tests{suffix}.<locals>.test_seeded_reset_deterministic"
    )
    test_random_actions_safe.__qualname__ = (
        f"register_env_invariant_tests{suffix}.<locals>.test_random_actions_safe"
    )

    return test_obs_or_reward, test_seeded_reset_deterministic, test_random_actions_safe
