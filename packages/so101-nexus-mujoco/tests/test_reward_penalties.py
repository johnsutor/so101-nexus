"""Reward-penalty wiring tests for MuJoCo primitive environments.

Covers the invariant that default configs (penalty coefficients zero) leave the
numeric reward unchanged, and that nonzero penalties strictly lower the reward
for an identical trajectory.
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import numpy as np
import pytest

import so101_nexus_mujoco  # noqa: F401 - registers envs
from so101_nexus_core.config import (
    LookAtConfig,
    MoveConfig,
    ReachConfig,
    RewardConfig,
)

_PRIMITIVES = [
    ("MuJoCoReach-v1", ReachConfig),
    ("MuJoCoMove-v1", MoveConfig),
    ("MuJoCoLookAt-v1", LookAtConfig),
]


def _run_trajectory(env_id, config, seed, actions):
    env = gym.make(env_id, config=config)
    try:
        env.reset(seed=seed)
        rewards = []
        infos = []
        for action in actions:
            _, reward, _, _, info = env.step(action)
            rewards.append(float(reward))
            infos.append(info)
        return rewards, infos
    finally:
        env.close()


@pytest.mark.parametrize(("env_id", "config_cls"), _PRIMITIVES)
def test_info_carries_penalty_norms(env_id, config_cls):
    env = gym.make(env_id, config=config_cls())
    try:
        env.reset(seed=0)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert "energy_norm" in info
        assert "action_delta_norm" in info
        # First step after reset has no previous action: delta is exactly 0.
        assert info["action_delta_norm"] == 0.0
    finally:
        env.close()


@pytest.mark.parametrize(("env_id", "config_cls"), _PRIMITIVES)
def test_first_step_delta_is_zero_then_positive(env_id, config_cls):
    env = gym.make(env_id, config=config_cls())
    try:
        env.reset(seed=1)
        rng = np.random.default_rng(1)
        a0 = env.action_space.sample()
        _, _, _, _, info0 = env.step(a0)
        assert info0["action_delta_norm"] == 0.0
        a1 = a0 + rng.uniform(-0.01, 0.01, size=a0.shape).astype(a0.dtype)
        a1 = np.clip(a1, env.action_space.low, env.action_space.high)
        _, _, _, _, info1 = env.step(a1)
        assert info1["action_delta_norm"] > 0.0
    finally:
        env.close()


@pytest.mark.parametrize(("env_id", "config_cls"), _PRIMITIVES)
def test_default_config_reward_has_no_penalty(env_id, config_cls):
    """Default config (zero penalties) must equal a pure no-penalty config."""
    probe = gym.make(env_id, config=config_cls())
    probe.reset(seed=7)
    actions = [probe.action_space.sample() for _ in range(4)]
    probe.close()

    default_rewards, _ = _run_trajectory(env_id, config_cls(), seed=7, actions=actions)
    explicit_zero = config_cls(reward=RewardConfig(action_delta_penalty=0.0, energy_penalty=0.0))
    zero_rewards, _ = _run_trajectory(env_id, explicit_zero, seed=7, actions=actions)
    assert default_rewards == pytest.approx(zero_rewards)


@pytest.mark.parametrize(("env_id", "config_cls"), _PRIMITIVES)
def test_nonzero_penalty_lowers_reward(env_id, config_cls):
    """A nonzero energy/action-delta penalty strictly lowers reward when norms are positive."""
    probe = gym.make(env_id, config=config_cls())
    probe.reset(seed=11)
    actions = [probe.action_space.sample() for _ in range(4)]
    probe.close()

    default_rewards, default_infos = _run_trajectory(env_id, config_cls(), seed=11, actions=actions)
    penalized = config_cls(reward=RewardConfig(action_delta_penalty=0.5, energy_penalty=0.5))
    pen_rewards, pen_infos = _run_trajectory(env_id, penalized, seed=11, actions=actions)

    # On steps where at least one norm is positive, the penalty must lower reward.
    lowered_any = False
    for i, (d, p) in enumerate(zip(default_rewards, pen_rewards, strict=True)):
        norms = default_infos[i]["energy_norm"] + default_infos[i]["action_delta_norm"]
        if norms > 0:
            assert p < d
            lowered_any = True
        else:
            assert p == pytest.approx(d)
    assert lowered_any
