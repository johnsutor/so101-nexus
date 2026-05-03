"""Property-based invariants for every MuJoCo SO101-Nexus environment."""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import so101_nexus_mujoco  # noqa: F401
from so101_nexus_core.testing.invariants import register_env_invariant_tests

ENV_IDS = [
    "MuJoCoReach-v1",
    "MuJoCoLookAt-v1",
    "MuJoCoMove-v1",
    "MuJoCoPickLift-v1",
    "MuJoCoPickAndPlace-v1",
]

(
    test_obs_in_space_and_reward_finite,
    test_seeded_reset_is_deterministic,
    test_random_actions_never_crash,
) = register_env_invariant_tests(
    ENV_IDS, max_examples=20, check_obs_in_space=True, db_namespace="mujoco"
)
