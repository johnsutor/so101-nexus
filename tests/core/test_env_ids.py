"""Unit + property tests for so101_nexus.env_ids."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from so101_nexus.env_ids import all_registered_env_ids, env_ids_for_backend

pytest.importorskip("gymnasium")


@pytest.fixture(autouse=True)
def _register_backend(monkeypatch):
    """Ensure a deterministic set of registered env ids for every test."""
    import gymnasium as gym

    fake_ids = [
        "MuJoCoReach-v1",
        "MuJoCoPickLift-v1",
        "CartPole-v1",  # unrelated; must be filtered out
    ]

    class _FakeRegistry(dict):
        def __iter__(self):
            return iter(fake_ids)

    monkeypatch.setattr(gym.envs, "registry", _FakeRegistry())


def test_all_registered_lists_mujoco_prefix():
    ids = all_registered_env_ids()
    assert "MuJoCoReach-v1" in ids
    assert "MuJoCoPickLift-v1" in ids
    assert "CartPole-v1" not in ids


def test_env_ids_for_backend_mujoco():
    assert env_ids_for_backend("mujoco") == [
        "MuJoCoReach-v1",
        "MuJoCoPickLift-v1",
    ]


def test_env_ids_for_backend_none_returns_all():
    assert env_ids_for_backend(None) == all_registered_env_ids()


@given(backend=st.sampled_from([None, "mujoco"]))
@settings(max_examples=30)
def test_filter_is_idempotent(backend):
    ids = env_ids_for_backend(backend)
    subset = [i for i in ids if i in all_registered_env_ids()]
    assert subset == ids


def test_warp_backend_prefix_registered():
    from so101_nexus.env_ids import _BACKEND_PREFIXES

    assert _BACKEND_PREFIXES["warp"] == "Warp"


def test_registered_ids_scan_covers_all_backend_prefixes(monkeypatch):
    import gymnasium as gym

    from so101_nexus import env_ids

    fake = {}
    for env_id in ("MuJoCoReach-v1", "WarpReach-v1", "CartPole-v1"):
        fake[env_id] = object()
    monkeypatch.setattr(gym.envs, "registry", fake)

    ids = env_ids._registered_so101_env_ids()
    assert "MuJoCoReach-v1" in ids
    assert "WarpReach-v1" in ids
    assert "CartPole-v1" not in ids
    assert env_ids.env_ids_for_backend("warp") == ["WarpReach-v1"]
    assert env_ids.env_ids_for_backend("mujoco") == ["MuJoCoReach-v1"]
