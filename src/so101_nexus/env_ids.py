"""Helpers for SO101-Nexus Gymnasium environment IDs."""

from __future__ import annotations

from typing import Literal

Backend = Literal["mujoco", "warp"]

_BACKEND_PREFIXES: dict[Backend, str] = {
    "mujoco": "MuJoCo",
    "warp": "Warp",
}


def _registered_so101_env_ids() -> list[str]:
    """Return registered SO101-Nexus env ids in registration order.

    Matches any known backend prefix so both ``MuJoCo*`` and ``Warp*`` ids are
    discovered once their backend module has been imported.
    """
    import gymnasium as gym

    prefixes = tuple(_BACKEND_PREFIXES.values())
    return [env_id for env_id in gym.envs.registry if env_id.startswith(prefixes)]


def all_registered_env_ids() -> list[str]:
    """Return all registered SO101-Nexus environment IDs.

    The list is sourced from ``gymnasium.envs.registry``, so the calling
    process must already have imported the backend it cares about
    (``import so101_nexus.mujoco`` and/or ``import so101_nexus.warp``) before
    calling this.
    """
    return _registered_so101_env_ids()


def env_ids_for_backend(backend: Backend | None) -> list[str]:
    """Return env ids for *backend* (``"mujoco"`` or ``"warp"``), or all if ``None``."""
    ids = _registered_so101_env_ids()
    if backend is None:
        return ids
    prefix = _BACKEND_PREFIXES[backend]
    return [env_id for env_id in ids if env_id.startswith(prefix)]


def backend_for_env_id(env_id: str) -> Backend:
    """Return the backend that owns ``env_id``, resolved from its id prefix.

    Parameters
    ----------
    env_id
        A SO101-Nexus Gymnasium id such as ``"MuJoCoPickLift-v1"``.

    Returns
    -------
    Backend
        ``"mujoco"`` or ``"warp"``.

    Raises
    ------
    ValueError
        If ``env_id`` carries no known backend prefix.
    """
    for backend, prefix in _BACKEND_PREFIXES.items():
        if env_id.startswith(prefix):
            return backend
    known = ", ".join(_BACKEND_PREFIXES.values())
    raise ValueError(f"{env_id!r} does not start with a known backend prefix ({known})")
