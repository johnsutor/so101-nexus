"""Helpers for SO101-Nexus Gymnasium environment IDs."""

from __future__ import annotations

from typing import Literal

Backend = Literal["mujoco", "maniskill"]

_BACKEND_PREFIXES: dict[Backend, str] = {
    "mujoco": "MuJoCo",
    "maniskill": "ManiSkill",
}


def _registered_so101_env_ids() -> list[str]:
    """Return registered ``MuJoCo*`` and ``ManiSkill*`` env ids in registration order."""
    import gymnasium as gym

    return [env_id for env_id in gym.envs.registry if env_id.startswith(("MuJoCo", "ManiSkill"))]


def all_registered_env_ids() -> list[str]:
    """Return all registered SO101-Nexus environment IDs.

    The list is sourced from ``gymnasium.envs.registry``, so the calling
    process must already have imported whichever backend(s) it cares about
    (``import so101_nexus_mujoco`` and/or ``import so101_nexus_maniskill``)
    before calling this.
    """
    return _registered_so101_env_ids()


def env_ids_for_backend(backend: Backend | None) -> list[str]:
    """Return env ids for *backend* (``"mujoco"`` or ``"maniskill"``), or all if ``None``."""
    ids = _registered_so101_env_ids()
    if backend is None:
        return ids
    prefix = _BACKEND_PREFIXES[backend]
    return [env_id for env_id in ids if env_id.startswith(prefix)]
