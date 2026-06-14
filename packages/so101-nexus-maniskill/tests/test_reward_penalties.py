"""Reward-penalty wiring tests for ManiSkill primitive environments.

Kept at ``num_envs=1`` so they run on CPU-only CI (vectorized envs need a GPU).
Covers the invariant that default configs (zero penalty coefficients) leave the
numeric reward unchanged, that nonzero penalties strictly lower reward, and that
both penalty norms appear in the step info dict.
"""

from __future__ import annotations

import gymnasium as gym
import pytest
import torch

import so101_nexus_maniskill  # noqa: F401 - registers envs
from so101_nexus_core.config import LookAtConfig, MoveConfig, ReachConfig, RewardConfig

BASE_KWARGS = {"obs_mode": "state", "num_envs": 1, "render_mode": None}

_PRIMITIVES = [
    ("ManiSkillReachSO101-v1", ReachConfig),
    ("ManiSkillMoveSO101-v1", MoveConfig),
    ("ManiSkillLookAtSO101-v1", LookAtConfig),
]


@pytest.mark.parametrize(("env_id", "config_cls"), _PRIMITIVES)
def test_info_carries_penalty_norms(env_id, config_cls):
    env = gym.make(env_id, config=config_cls(), **BASE_KWARGS)
    try:
        env.reset(seed=0)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert "energy_norm" in info
        assert "action_delta_norm" in info
        # First step after reset: no previous action, delta is exactly 0.
        assert torch.all(info["action_delta_norm"] == 0.0)
    finally:
        env.close()


@pytest.mark.parametrize(("env_id", "config_cls"), _PRIMITIVES)
def test_default_config_reward_unchanged(env_id, config_cls):
    """Default config reward equals an explicit zero-penalty config reward."""
    # Capture a fixed action sequence once so both rollouts replay the same
    # trajectory (action_space.sample uses its own RNG, not the torch seed).
    probe = gym.make(env_id, config=config_cls(), **BASE_KWARGS)
    probe.action_space.seed(3)
    actions = [probe.action_space.sample() for _ in range(4)]
    probe.close()

    def rollout(config):
        env = gym.make(env_id, config=config, **BASE_KWARGS)
        try:
            env.reset(seed=3)
            rewards = []
            for action in actions:
                _, reward, _, _, _ = env.step(action)
                rewards.append(reward.clone())
            return rewards
        finally:
            env.close()

    default_rewards = rollout(config_cls())
    zero_rewards = rollout(
        config_cls(reward=RewardConfig(action_delta_penalty=0.0, energy_penalty=0.0))
    )
    for d, z in zip(default_rewards, zero_rewards, strict=True):
        torch.testing.assert_close(d, z)


@pytest.mark.parametrize(("env_id", "config_cls"), _PRIMITIVES)
def test_nonzero_penalty_lowers_reward(env_id, config_cls):
    """A nonzero penalty strictly lowers reward on steps with positive norms."""

    probe = gym.make(env_id, config=config_cls(), **BASE_KWARGS)
    probe.action_space.seed(5)
    actions = [probe.action_space.sample() for _ in range(4)]
    probe.close()

    def rollout(config):
        env = gym.make(env_id, config=config, **BASE_KWARGS)
        try:
            env.reset(seed=5)
            out = []
            for action in actions:
                _, reward, _, _, info = env.step(action)
                out.append(
                    (reward.clone(), info["energy_norm"].clone(), info["action_delta_norm"].clone())
                )
            return out
        finally:
            env.close()

    default = rollout(config_cls())
    penalized = rollout(
        config_cls(reward=RewardConfig(action_delta_penalty=0.5, energy_penalty=0.5))
    )

    lowered_any = False
    for (d_r, e, a), (p_r, _, _) in zip(default, penalized, strict=True):
        norms = (e + a).sum().item()
        if norms > 0:
            assert torch.all(p_r < d_r)
            lowered_any = True
        else:
            torch.testing.assert_close(p_r, d_r)
    assert lowered_any
