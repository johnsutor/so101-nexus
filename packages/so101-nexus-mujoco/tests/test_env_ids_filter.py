"""Tests for backend-filtered env id helpers (mujoco side).

These live in the mujoco package because they need a real backend
imported into ``gymnasium.envs.registry`` to assert against.
"""

from __future__ import annotations

import so101_nexus_mujoco  # noqa: F401 — registers MuJoCo gym envs
from so101_nexus_core.env_ids import all_registered_env_ids, env_ids_for_backend


def test_mujoco_filter_returns_only_mujoco_envs() -> None:
    ids = env_ids_for_backend("mujoco")
    assert len(ids) >= 5
    assert all(i.startswith("MuJoCo") for i in ids)


def test_mujoco_filter_includes_pick_and_place() -> None:
    ids = env_ids_for_backend("mujoco")
    assert "MuJoCoPickAndPlace-v1" in ids


def test_none_backend_includes_mujoco_envs() -> None:
    ids = env_ids_for_backend(None)
    assert any(i.startswith("MuJoCo") for i in ids)
    assert ids == all_registered_env_ids()
