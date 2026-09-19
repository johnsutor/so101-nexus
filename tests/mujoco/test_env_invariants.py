"""Property-based invariants for every MuJoCo SO101-Nexus environment."""

from __future__ import annotations

import subprocess
import sys

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
    "MuJoCoPickAndPlace-v2",
    "MuJoCoPickReturn-v1",
    "MuJoCoStackCube-v1",
]


def _make_from_factory(env_factory, env_id):
    stem, raw_version = env_id.rsplit("-v", maxsplit=1)
    return env_factory(task=stem.removeprefix("MuJoCo"), version=int(raw_version))


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_seeded_rollout_replays_after_dirty_episode(env_id):
    import gymnasium as gym

    import so101_nexus.mujoco  # noqa: F401

    env = gym.make(env_id)
    try:
        env.reset(seed=19)
        env.action_space.seed(23)
        actions = [env.action_space.sample() for _ in range(10)]
        expected = [env.step(action)[:4] for action in actions]

        env.reset(seed=97)
        for _ in range(7):
            env.step(env.action_space.sample())

        env.reset(seed=19)
        actual = [env.step(action)[:4] for action in actions]
        for expected_step, actual_step in zip(expected, actual, strict=True):
            np.testing.assert_array_equal(expected_step[0], actual_step[0])
            assert expected_step[1:] == actual_step[1:]
    finally:
        env.close()


def test_seeded_rollouts_match_across_fresh_processes():
    script = f"""
import hashlib
import pickle
import gymnasium as gym
import numpy as np
import so101_nexus.mujoco

env_ids = {ENV_IDS!r}
fingerprints = []
for env_id in env_ids:
    env = gym.make(env_id)
    try:
        observation, info = env.reset(seed=31)
        rng = np.random.default_rng(37)
        trajectory = [(observation, info, env.unwrapped.task_description)]
        for _ in range(10):
            action = rng.uniform(env.action_space.low, env.action_space.high).astype(np.float32)
            trajectory.append(env.step(action))
        fingerprints.append(trajectory)
    finally:
        env.close()
print(hashlib.sha256(pickle.dumps(fingerprints)).hexdigest())
"""

    def fingerprint() -> str:
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    assert fingerprint() == fingerprint()


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_obs_always_in_observation_space(env_id, env_factory):
    """Observation returned by reset/step always belongs to ``observation_space``."""
    env = _make_from_factory(env_factory, env_id)

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
    env = _make_from_factory(env_factory, env_id)

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
