"""Property-based invariants for every ManiSkill SO101-Nexus environment."""

from __future__ import annotations

import so101_nexus_maniskill  # noqa: F401
from so101_nexus_core.testing.invariants import register_env_invariant_tests

BASE_KWARGS = {"obs_mode": "state", "num_envs": 1, "render_mode": None}

ENV_IDS = [
    "ManiSkillReachSO100-v1",
    "ManiSkillReachSO101-v1",
    "ManiSkillLookAtSO100-v1",
    "ManiSkillLookAtSO101-v1",
    "ManiSkillMoveSO100-v1",
    "ManiSkillMoveSO101-v1",
    "ManiSkillPickLiftSO100-v1",
    "ManiSkillPickLiftSO101-v1",
    "ManiSkillPickAndPlaceSO100-v1",
    "ManiSkillPickAndPlaceSO101-v1",
]

(
    test_reward_is_finite,
    test_seeded_reset_is_deterministic,
    _test_random_actions_unused,
) = register_env_invariant_tests(
    ENV_IDS,
    base_kwargs=BASE_KWARGS,
    max_examples=10,
    check_obs_in_space=False,
    db_namespace="maniskill",
)
# ManiSkill historically did not run the random-actions test (tensor batched
# envs handle that via test_envs.py). Drop the unused third test.
del _test_random_actions_unused
