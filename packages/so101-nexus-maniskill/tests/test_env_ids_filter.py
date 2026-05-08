"""Tests for backend-filtered env id helpers (maniskill side)."""

from __future__ import annotations

import so101_nexus_maniskill  # noqa: F401 — registers ManiSkill gym envs
from so101_nexus_core.testing.env_id_filter import run_env_id_filter_contract


def test_maniskill_filter_contract() -> None:
    run_env_id_filter_contract(
        backend="maniskill",
        prefix="ManiSkill",
        min_count=8,
        required_ids=("ManiSkillPickAndPlaceSO100-v1", "ManiSkillPickAndPlaceSO101-v1"),
    )
