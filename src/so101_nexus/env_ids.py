"""Helpers for SO101-Nexus Gymnasium environment IDs."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from gymnasium.vector import VectorEnv

Backend = Literal["mujoco", "warp"]

_BACKEND_PREFIXES: dict[Backend, str] = {
    "mujoco": "MuJoCo",
    "warp": "Warp",
}

_TASKS = (
    ("Touch-v1", "touch_env", "Touch", 512),
    ("LookAt-v1", "look_at_env", "LookAt", 256),
    ("Move-v1", "move_env", "Move", 256),
    ("PickLift-v1", "pick_env", "PickLift", 1024),
    ("PickReturn-v1", "pick_return", "PickReturn", 1024),
    ("PickAndPlace-v1", "pick_and_place", "PickAndPlace", 1024),
    ("PickAndPlace-v2", "pick_and_place", "PickAndPlaceV2", 1024),
    ("StackCube-v1", "stack_cube", "StackCube", 1024),
)


def _register_backend(backend: Backend) -> None:
    """Register one backend's task inventory without importing its simulator."""
    import gymnasium as gym

    # Preserve discovery order for callers choosing the first available task.
    tasks = _TASKS[3:] + _TASKS[:3] if backend == "mujoco" else _TASKS
    for task_id, module, cls, limit in tasks:
        env_id = _BACKEND_PREFIXES[backend] + task_id
        scalar_entry = f"so101_nexus.mujoco.{module}:{cls}Env"
        if backend == "mujoco":
            gym.register(
                id=env_id,
                entry_point=scalar_entry,
                vector_entry_point=partial(_make_mujoco_vector_env, task_entry_point=scalar_entry),
                max_episode_steps=limit,
            )
        else:
            gym.register(
                id=env_id,
                vector_entry_point=f"so101_nexus.warp.{module}:Warp{cls}VectorEnv",
                max_episode_steps=limit,
            )


def _make_mujoco_vector_env(**kwargs: Any) -> VectorEnv:
    """Load native CPU batching only when a vector environment is constructed."""
    from so101_nexus.mujoco.vector_env import MuJoCoVectorEnv

    return MuJoCoVectorEnv(**kwargs)


def _registered_so101_env_ids() -> list[str]:
    """Return registered SO101-Nexus env ids in registration order.

    Matches any known backend prefix so ``MuJoCo*`` and ``Warp*`` ids are
    discovered once their backend module has been imported.
    """
    import gymnasium as gym

    prefixes = tuple(_BACKEND_PREFIXES.values())
    return [env_id for env_id in gym.envs.registry if env_id.startswith(prefixes)]


def all_registered_env_ids() -> list[str]:
    """Return all registered SO101-Nexus environment IDs.

    The list is sourced from ``gymnasium.envs.registry``, so the calling
    process must already have imported the backend it cares about
    (``so101_nexus.mujoco`` or ``so101_nexus.warp``)
    before calling this.
    """
    return _registered_so101_env_ids()


def env_ids_for_backend(backend: Backend | None) -> list[str]:
    """Return env ids for the selected backend, or all registered ids if ``None``."""
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
