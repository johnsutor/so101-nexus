"""Tests for backend-filtered env id helpers (maniskill side)."""

from __future__ import annotations

import so101_nexus_maniskill  # noqa: F401 — registers ManiSkill gym envs
from so101_nexus_core.env_ids import env_ids_for_backend


def test_maniskill_filter_returns_only_maniskill_envs() -> None:
    ids = env_ids_for_backend("maniskill")
    assert len(ids) >= 8
    assert all(i.startswith("ManiSkill") for i in ids)


def test_maniskill_filter_includes_pick_and_place() -> None:
    ids = env_ids_for_backend("maniskill")
    assert "ManiSkillPickAndPlaceSO100-v1" in ids
    assert "ManiSkillPickAndPlaceSO101-v1" in ids
