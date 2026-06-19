"""Helpers for SO101-Nexus Gymnasium environment IDs."""

from __future__ import annotations

from typing import Literal

Backend = Literal["mujoco"]

_BACKEND_PREFIXES: dict[Backend, str] = {
    "mujoco": "MuJoCo",
}


def _registered_so101_env_ids() -> list[str]:
    """Return registered ``MuJoCo*`` env ids in registration order."""
    import gymnasium as gym

    return [env_id for env_id in gym.envs.registry if env_id.startswith("MuJoCo")]


def all_registered_env_ids() -> list[str]:
    """Return all registered SO101-Nexus environment IDs.

    The list is sourced from ``gymnasium.envs.registry``, so the calling
    process must already have imported the backend it cares about
    (``import so101_nexus.mujoco``) before calling this.
    """
    return _registered_so101_env_ids()


def env_ids_for_backend(backend: Backend | None) -> list[str]:
    """Return env ids for *backend* (``"mujoco"``), or all if ``None``."""
    ids = _registered_so101_env_ids()
    if backend is None:
        return ids
    prefix = _BACKEND_PREFIXES[backend]
    return [env_id for env_id in ids if env_id.startswith(prefix)]
