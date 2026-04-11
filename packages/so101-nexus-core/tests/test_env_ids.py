"""Tests for env id helpers."""

from __future__ import annotations

from so101_nexus_core.env_ids import all_registered_env_ids, env_ids_for_backend


def test_all_registered_env_ids_contains_both_backends() -> None:
    ids = all_registered_env_ids()
    assert any(i.startswith("MuJoCo") for i in ids)
    assert any(i.startswith("ManiSkill") for i in ids)


def test_env_ids_for_backend_mujoco_only() -> None:
    ids = env_ids_for_backend("mujoco")
    assert len(ids) > 0
    assert all(i.startswith("MuJoCo") for i in ids)
    assert not any(i.startswith("ManiSkill") for i in ids)


def test_env_ids_for_backend_maniskill_only() -> None:
    ids = env_ids_for_backend("maniskill")
    assert len(ids) > 0
    assert all(i.startswith("ManiSkill") for i in ids)
    assert not any(i.startswith("MuJoCo") for i in ids)


def test_env_ids_for_backend_none_returns_all() -> None:
    assert env_ids_for_backend(None) == all_registered_env_ids()
