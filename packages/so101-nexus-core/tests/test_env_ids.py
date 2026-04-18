"""Unit + property tests for so101_nexus_core.env_ids."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from so101_nexus_core.env_ids import all_registered_env_ids, env_ids_for_backend


@pytest.fixture(autouse=True)
def _register_both_backends(monkeypatch):
    """Ensure a deterministic set of registered env ids for every test."""
    import gymnasium as gym

    fake_ids = [
        "MuJoCoReach-v1",
        "MuJoCoPickLift-v1",
        "ManiSkillReachSO100-v1",
        "ManiSkillPickLiftSO101-v1",
        "CartPole-v1",  # unrelated; must be filtered out
    ]

    class _FakeRegistry(dict):
        def __iter__(self):
            return iter(fake_ids)

    monkeypatch.setattr(gym.envs, "registry", _FakeRegistry())


def test_all_registered_lists_both_prefixes():
    ids = all_registered_env_ids()
    assert "MuJoCoReach-v1" in ids
    assert "ManiSkillReachSO100-v1" in ids
    assert "CartPole-v1" not in ids


def test_env_ids_for_backend_mujoco():
    assert env_ids_for_backend("mujoco") == [
        "MuJoCoReach-v1",
        "MuJoCoPickLift-v1",
    ]


def test_env_ids_for_backend_maniskill():
    assert env_ids_for_backend("maniskill") == [
        "ManiSkillReachSO100-v1",
        "ManiSkillPickLiftSO101-v1",
    ]


def test_env_ids_for_backend_none_returns_all():
    assert env_ids_for_backend(None) == all_registered_env_ids()


@given(backend=st.sampled_from([None, "mujoco", "maniskill"]))
@settings(max_examples=30)
def test_filter_is_idempotent(backend):
    ids = env_ids_for_backend(backend)
    subset = [i for i in ids if i in all_registered_env_ids()]
    assert subset == ids
