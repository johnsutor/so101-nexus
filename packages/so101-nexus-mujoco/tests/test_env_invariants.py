"""Property-based invariants for every MuJoCo SO101-Nexus environment."""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import so101_nexus_mujoco  # noqa: F401
from so101_nexus_core.testing.invariants import (
    assert_obs_always_in_observation_space,
    assert_random_actions_never_crash,
    assert_seeded_reset_is_deterministic,
)

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
    assert_obs_always_in_observation_space(env_id, seed)


@pytest.mark.parametrize("env_id", ENV_IDS)
@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_seeded_reset_is_deterministic(env_id, seed):
    assert_seeded_reset_is_deterministic(env_id, seed)


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_random_actions_never_crash(env_id):
    assert_random_actions_never_crash(env_id, steps=20)
