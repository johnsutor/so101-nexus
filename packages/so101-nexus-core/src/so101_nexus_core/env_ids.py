"""Helpers for SO101-Nexus Gymnasium environment IDs."""

from __future__ import annotations

from typing import Literal

Backend = Literal["mujoco", "maniskill"]

_MUJOCO_ENV_IDS: tuple[str, ...] = (
    "MuJoCoPickLift-v1",
    "MuJoCoPickAndPlace-v1",
    "MuJoCoReach-v1",
    "MuJoCoLookAt-v1",
    "MuJoCoMove-v1",
)

_MANISKILL_ENV_IDS: tuple[str, ...] = (
    "ManiSkillPickLiftSO100-v1",
    "ManiSkillPickLiftSO101-v1",
    "ManiSkillReachSO100-v1",
    "ManiSkillReachSO101-v1",
    "ManiSkillLookAtSO100-v1",
    "ManiSkillLookAtSO101-v1",
    "ManiSkillMoveSO100-v1",
    "ManiSkillMoveSO101-v1",
)


def all_registered_env_ids() -> list[str]:
    """Return a list of all registered SO101-Nexus environment IDs."""
    return [*_MUJOCO_ENV_IDS, *_MANISKILL_ENV_IDS]


def env_ids_for_backend(backend: Backend | None) -> list[str]:
    """Return env ids for *backend* (``"mujoco"`` or ``"maniskill"``), or all if ``None``."""
    if backend == "mujoco":
        return list(_MUJOCO_ENV_IDS)
    if backend == "maniskill":
        return list(_MANISKILL_ENV_IDS)
    return all_registered_env_ids()
