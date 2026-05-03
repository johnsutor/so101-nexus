"""Tests for backend-filtered env id helpers (mujoco side)."""

from __future__ import annotations

import so101_nexus_mujoco  # noqa: F401 — registers MuJoCo gym envs
from so101_nexus_core.testing.env_id_filter import (
    assert_none_backend_includes,
    run_env_id_filter_contract,
)


def test_mujoco_filter_returns_only_mujoco_envs():
    run_env_id_filter_contract(
        "mujoco",
        prefix="MuJoCo",
        must_include=["MuJoCoPickAndPlace-v1"],
        min_count=5,
    )


def test_none_backend_includes_mujoco_envs():
    assert_none_backend_includes("MuJoCo")
