"""Tests for backend-filtered env id helpers (mujoco side).

These live in the mujoco package because they need a real backend
imported into ``gymnasium.envs.registry`` to assert against.
"""

from __future__ import annotations

import so101_nexus_mujoco  # noqa: F401 - registers MuJoCo gym envs
from so101_nexus_core.testing.env_id_filter import run_env_id_filter_contract


def test_mujoco_filter_contract() -> None:
    run_env_id_filter_contract(
        backend="mujoco",
        prefix="MuJoCo",
        min_count=5,
        required_ids=("MuJoCoPickAndPlace-v1",),
    )
