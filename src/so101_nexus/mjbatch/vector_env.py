"""MuJoCo task semantics with optional native CPU batch stepping."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium.envs.registration import load_env_creator
from gymnasium.vector import AutoresetMode, SyncVectorEnv

from so101_nexus.mujoco.base_env import SO101NexusMuJoCoBaseEnv

if TYPE_CHECKING:
    from collections.abc import Callable

    from so101_nexus.config import EnvironmentConfig

_STATE = mujoco.mjtState.mjSTATE_INTEGRATION
_MODEL_FIELDS = (
    "geom_contype",
    "geom_conaffinity",
    "geom_rgba",
    "mat_rgba",
    "body_pos",
    "site_pos",
    "cam_pos",
    "cam_quat",
    "cam_fovy",
)


class _CompletedTask(gym.Wrapper):
    """Preserve Gymnasium wrappers while evaluating physics advanced by the batch."""

    env: SO101NexusMuJoCoBaseEnv

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict]:
        return self.env._finish_step(action)


class MJBatchVectorEnv(SyncVectorEnv):
    """Batch CPU physics while reusing MuJoCo tasks, controls, and observations.

    Parameters
    ----------
    task_entry_point : str
        Registered MuJoCo task class reused by this vector environment.
    num_envs : int
        Number of independent simulations sharing a scene topology.
    num_threads : int
        Native worker count, or zero to select automatically.
    max_episode_steps : int
        Time limit per simulation. A negative value disables the limit.
    autoreset_mode : AutoresetMode or str
        Gymnasium reset timing, including masked explicit resets.
    config : EnvironmentConfig, optional
        Task configuration copied independently for each simulation.
    **kwargs
        MuJoCo task constructor options, including controls and rendering.

    Notes
    -----
    The final substep runs on each task's MjData to retain MuJoCo's contact
    and observation timing. Tasks with custom substep hooks use their scalar
    physics path, including PickAndPlace v2's continuous support evaluator.
    Resets and camera rendering also use the existing MuJoCo implementation.
    """

    def __init__(
        self,
        *,
        task_entry_point: str,
        num_envs: int = 1,
        num_threads: int = 0,
        max_episode_steps: int = 1024,
        autoreset_mode: AutoresetMode | str = AutoresetMode.NEXT_STEP,
        config: EnvironmentConfig | None = None,
        _task_wrapper: Callable[[gym.Env], gym.Env] | None = None,
        **kwargs: Any,
    ) -> None:
        if num_envs < 1 or num_threads < 0:
            raise ValueError("num_envs must be positive and num_threads must be nonnegative")
        try:
            from mjbatch import Batch
        except ImportError as exc:
            raise ImportError(
                'Install the CPU batch backend with pip install "so101-nexus[mjbatch]"'
            ) from exc
        creator = load_env_creator(task_entry_point)

        def make_task() -> gym.Env:
            raw_task = creator(config=deepcopy(config), **kwargs)
            if not isinstance(raw_task, SO101NexusMuJoCoBaseEnv):
                raise TypeError("mjbatch requires a MuJoCo task entry point")
            task: gym.Env = _CompletedTask(raw_task)
            if max_episode_steps > 0:
                task = gym.wrappers.TimeLimit(task, max_episode_steps)
            return task if _task_wrapper is None else _task_wrapper(task)

        super().__init__([make_task] * num_envs, autoreset_mode=autoreset_mode)
        self._tasks: list[SO101NexusMuJoCoBaseEnv] = []
        for env in self.envs:
            raw_task = env.unwrapped
            assert isinstance(raw_task, SO101NexusMuJoCoBaseEnv)
            self._tasks.append(raw_task)
        try:
            self.batch = Batch(self._tasks[0].model, num_sims=num_envs, num_threads=num_threads)
            self._states = self.batch.bind("state")
            self._model_fields = {name: self.batch.expand(name) for name in _MODEL_FIELDS}
        except Exception:
            self.close()
            raise
        self._native_substeps = (
            type(self._tasks[0])._advance_physics is SO101NexusMuJoCoBaseEnv._advance_physics
        )

    @property
    def task_descriptions(self) -> list[str]:
        """Return the current instruction for each simulation."""
        return [task.task_description for task in self._tasks]

    @property
    def task_description(self) -> str:
        """Return the shared instruction, or the task family when worlds differ."""
        descriptions = self.task_descriptions
        return descriptions[0] if len(set(descriptions)) == 1 else "SO-101 manipulation tasks."

    def step(self, actions: np.ndarray) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, dict]:
        """Advance active simulations and apply Gymnasium's autoreset contract."""
        actions = np.asarray(actions)
        if actions.shape != self.action_space.shape:
            raise ValueError(f"actions shape {actions.shape} != expected {self.action_space.shape}")
        if self.autoreset_mode == AutoresetMode.DISABLED:
            assert not self._autoreset_envs.any(), "Reset completed simulations before stepping"
        active = (
            ~self._autoreset_envs
            if self.autoreset_mode == AutoresetMode.NEXT_STEP
            else np.ones(self.num_envs, dtype=bool)
        )
        for i in np.flatnonzero(active):
            task = self._tasks[i]
            task.data.ctrl[task._actuator_ids] = task._action_to_ctrl(actions[i])
            mujoco.mj_getState(task.model, task.data, self._states[i], _STATE)
            for name, values in self._model_fields.items():
                values[i] = getattr(task.model, name)
        if self._native_substeps:
            substeps = self._tasks[0]._N_SUBSTEPS
            if active.any() and substeps > 1:
                self.batch.step(ids=active, nstep=substeps - 1)
            for i in np.flatnonzero(active):
                task = self._tasks[i]
                mujoco.mj_setState(task.model, task.data, self._states[i], _STATE)
                mujoco.mj_step(task.model, task.data)
        else:
            for i in np.flatnonzero(active):
                self._tasks[i]._advance_physics()
        return super().step(actions)

    def close_extras(self, **kwargs: Any) -> None:
        """Release task renderers and all native batch references."""
        super().close_extras(**kwargs)
        for name in ("_states", "_model_fields", "batch"):
            if hasattr(self, name):
                delattr(self, name)
