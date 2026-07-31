"""LeRobot EnvHub entry point for the SO101-Nexus environments.

LeRobot's EnvHub downloads a Python file from a Hugging Face Hub repository and
calls its ``make_env(n_envs, use_async_envs[, cfg])`` function (see
https://huggingface.co/docs/lerobot/en/envhub). The files published to the Hub
are thin shims over this module (``envhub/`` in the source repository), so the
loading logic ships, is versioned, and is tested with the library instead of
living only in a Hub repository.

Observations follow the gym-side convention LeRobot's
``lerobot.envs.utils.preprocess_observation`` consumes: ``agent_pos`` (the six
joint positions), ``environment_state`` (the full state vector) and ``pixels``
(camera name to HWC uint8 image). That is one layer below the
``observation.state`` / ``observation.images.*`` keys
``so101_nexus.processors.LeRobotEnvWrapper`` produces, which shape observations
for a policy rather than for a LeRobot environment consumer.

Units are the simulator's own: joint angles in radians, matching the gym action
space. Datasets recorded through ``so101_nexus.lerobot_adapter`` store LeRobot
motor units instead (degrees, with the gripper in ``RANGE_0_100``); convert with
``so101_nexus.dataset_row_to_sim_qpos``.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, cast

import gymnasium as gym
import numpy as np
from gymnasium.vector import AutoresetMode
from gymnasium.vector.utils import batch_space

from so101_nexus.env_ids import backend_for_env_id
from so101_nexus.observations import (
    JointPositions,
    OverheadCamera,
    WristCamera,
    component_slice,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from so101_nexus.config import EnvironmentConfig

DEFAULT_ENV_ID = "MuJoCoPickLift-v1"
"""Environment served when the caller names no task."""

STATE_OBS_TYPE = "state"
PIXELS_OBS_TYPE = "pixels_agent_pos"

_CAMERA_SUFFIX = "_camera"

# Every knob ``make_env`` understands, with its default. Options come from an
# ``EnvConfig`` (LeRobot's own fields, plus a free-form ``kwargs`` dict, matching
# how ``lerobot.envs.configs.IsaaclabArenaEnv`` passes hub-specific settings) and
# from direct keyword arguments, which the per-environment Hub shims use.
_OPTION_DEFAULTS: dict[str, Any] = {
    "env_id": DEFAULT_ENV_ID,
    "obs_type": STATE_OBS_TYPE,
    "observation_width": 224,
    "observation_height": 224,
    "episode_length": None,
    "control_mode": None,
    "render_mode": None,
    "disable_env_checker": True,
    "device": None,
    "config": None,
}

_CFG_FIELD_TO_OPTION = {
    "task": "env_id",
    "obs_type": "obs_type",
    "observation_width": "observation_width",
    "observation_height": "observation_height",
    "episode_length": "episode_length",
    "disable_env_checker": "disable_env_checker",
}


def make_env(
    n_envs: int = 1,
    use_async_envs: bool = False,
    cfg: Any = None,
    **overrides: Any,
) -> dict[str, dict[int, gym.vector.VectorEnv]]:
    """Build SO101-Nexus environments in the shape LeRobot's EnvHub expects.

    Parameters
    ----------
    n_envs
        Number of parallel environments.
    use_async_envs
        Use ``AsyncVectorEnv`` instead of ``SyncVectorEnv``. Ignored by the Warp
        backend, whose environments are natively batched in one process. LeRobot's
        own rollout indexes ``VectorEnv.envs``, which Gymnasium's ``AsyncVectorEnv``
        does not expose, so leave this off when LeRobot drives the environment.
    cfg
        Optional LeRobot ``EnvConfig``. ``task`` selects the environment id,
        ``obs_type`` picks ``"state"`` or ``"pixels_agent_pos"``,
        ``observation_width`` / ``observation_height`` size the cameras,
        ``episode_length`` overrides the registered step limit, and a ``kwargs``
        dict carries any of the keyword overrides below.
    **overrides
        Direct overrides for the same options, plus ``env_id``, ``control_mode``,
        ``render_mode``, ``device`` (Warp only) and ``config`` (a fully built
        ``EnvironmentConfig``, which takes precedence over ``obs_type`` and the
        camera resolution).

    Returns
    -------
    dict
        ``{env_id: {0: vector_env}}``, the mapping LeRobot normalizes hub
        results into.

    Raises
    ------
    ValueError
        If ``n_envs`` is below one, an option is unknown, or ``obs_type`` is
        neither ``"state"`` nor ``"pixels_agent_pos"``.
    """
    if n_envs < 1:
        raise ValueError(f"n_envs must be >= 1, got {n_envs}")
    options = _resolve_options(cfg, overrides)
    env_id = options["env_id"]
    backend = backend_for_env_id(env_id)
    importlib.import_module(f"so101_nexus.{backend}")  # registers the gym ids
    config = options["config"]
    if config is None:
        config = _task_config(env_id, options)
    if backend == "warp":
        env = _make_warp_env(env_id, n_envs, config, options)
    else:
        env = _make_mujoco_env(env_id, n_envs, use_async_envs, config, options)
    return {env_id: {0: env}}


def _resolve_options(cfg: Any, overrides: dict[str, Any]) -> dict[str, Any]:
    """Merge defaults, ``cfg`` fields, ``cfg.kwargs`` and keyword overrides."""
    options = dict(_OPTION_DEFAULTS)
    for field, option in _CFG_FIELD_TO_OPTION.items():
        value = getattr(cfg, field, None)
        if value is not None:
            options[option] = value
    for source in (getattr(cfg, "kwargs", None) or {}, overrides):
        unknown = sorted(set(source) - set(_OPTION_DEFAULTS))
        if unknown:
            known = ", ".join(sorted(_OPTION_DEFAULTS))
            raise ValueError(f"Unknown make_env options {unknown}; known options are {known}")
        options.update({key: value for key, value in source.items() if value is not None})
    if options["obs_type"] not in (STATE_OBS_TYPE, PIXELS_OBS_TYPE):
        raise ValueError(
            f"obs_type must be {STATE_OBS_TYPE!r} or {PIXELS_OBS_TYPE!r}, "
            f"got {options['obs_type']!r}"
        )
    return options


def _task_config(env_id: str, options: dict[str, Any]) -> EnvironmentConfig | None:
    """Return the task config for ``options``, or ``None`` to use the env default.

    The visual config keeps the task's default state components so
    ``info["privileged_state"]`` still carries the full ground truth: in
    ``obs_mode="visual"`` those components are exactly the privileged half of the
    asymmetric actor-critic split, and the observation itself is unaffected
    (visual mode's ``state`` entry is the joint positions whatever the list holds).
    """
    if options["obs_type"] == STATE_OBS_TYPE:
        return None
    width, height = options["observation_width"], options["observation_height"]
    config_cls = _default_config_cls(env_id)
    defaults = config_cls().observations or []
    return config_cls(
        obs_mode="visual",
        observations=[
            *defaults,
            WristCamera(width=width, height=height),
            OverheadCamera(width=width, height=height),
        ],
    )


def _default_config_cls(env_id: str) -> type[EnvironmentConfig]:
    """Return the config class of the env class registered under ``env_id``."""
    spec = gym.spec(env_id)
    target = str(spec.entry_point or spec.vector_entry_point)
    module_path, _, attribute = target.partition(":")
    return getattr(importlib.import_module(module_path), attribute).default_config_cls


def _make_mujoco_env(
    env_id: str,
    n_envs: int,
    use_async_envs: bool,
    config: EnvironmentConfig | None,
    options: dict[str, Any],
) -> gym.vector.VectorEnv:
    """Vectorize ``n_envs`` independent MuJoCo environments."""
    make_kwargs: dict[str, Any] = {
        "config": config,
        "control_mode": options["control_mode"],
        "max_episode_steps": options["episode_length"],
        "render_mode": options["render_mode"],
    }
    make_kwargs = {key: value for key, value in make_kwargs.items() if value is not None}
    make_kwargs["disable_env_checker"] = options["disable_env_checker"]

    def thunk() -> gym.Env:
        return EnvHubAdapter(gym.make(env_id, **make_kwargs))

    env_cls = gym.vector.AsyncVectorEnv if use_async_envs else gym.vector.SyncVectorEnv
    # SAME_STEP is what fills ``info["final_info"]``, where LeRobot's rollout
    # reads per-episode success, and it matches the Warp backend's autoreset.
    return env_cls([thunk] * n_envs, autoreset_mode=AutoresetMode.SAME_STEP)


def _make_warp_env(
    env_id: str,
    n_envs: int,
    config: EnvironmentConfig | None,
    options: dict[str, Any],
) -> gym.vector.VectorEnv:
    """Build one natively batched Warp environment holding ``n_envs`` worlds."""
    make_kwargs: dict[str, Any] = {
        "config": config,
        "control_mode": options["control_mode"],
        "max_episode_steps": options["episode_length"],
        "device": options["device"],
    }
    make_kwargs = {key: value for key, value in make_kwargs.items() if value is not None}
    return WarpEnvHubAdapter(gym.make_vec(env_id, num_envs=n_envs, **make_kwargs))


def _camera_key(name: str) -> str:
    """Return the ``pixels`` sub-key for an observation key such as ``wrist_camera``."""
    return name[: -len(_CAMERA_SUFFIX)]


def _joint_positions_slice(observations: Sequence[Any] | None) -> slice | None:
    """Return the flat-state slice holding the joint positions, if observed."""
    if not any(isinstance(component, JointPositions) for component in observations or ()):
        return None
    return component_slice(observations, JointPositions)


def _adapt_space(space: gym.Space, joint_slice: slice | None) -> gym.spaces.Dict:
    """Return the LeRobot-convention observation space for a SO101-Nexus space."""
    # A visual-mode env already splits proprioception out: its "state" entry is
    # the joint positions, so only the flat-vector branch needs joint_slice.
    if isinstance(space, gym.spaces.Dict):
        spaces: dict[str, gym.Space] = {"agent_pos": space["state"]}
        pixels = {
            _camera_key(name): sub
            for name, sub in space.spaces.items()
            if name.endswith(_CAMERA_SUFFIX)
        }
        if pixels:
            spaces["pixels"] = gym.spaces.Dict(pixels)
        return gym.spaces.Dict(spaces)
    if not isinstance(space, gym.spaces.Box):
        raise TypeError(f"expected a Box or Dict observation space, got {type(space).__name__}")
    spaces = {"environment_state": space}
    if joint_slice is not None:
        spaces["agent_pos"] = gym.spaces.Box(
            low=space.low[joint_slice], high=space.high[joint_slice], dtype=space.low.dtype
        )
    return gym.spaces.Dict(spaces)


def _adapt_observation(observation: Any, joint_slice: slice | None) -> dict[str, Any]:
    """Re-key one observation (batched or not) into the LeRobot gym convention."""
    if isinstance(observation, dict):
        adapted: dict[str, Any] = {"agent_pos": observation["state"]}
        pixels = {
            _camera_key(name): image
            for name, image in observation.items()
            if name.endswith(_CAMERA_SUFFIX)
        }
        if pixels:
            adapted["pixels"] = pixels
        return adapted
    adapted = {"environment_state": observation}
    if joint_slice is not None:
        adapted["agent_pos"] = observation[..., joint_slice]
    return adapted


class EnvHubAdapter(gym.Wrapper):
    """Present one SO101-Nexus environment through LeRobot's gym conventions.

    Three gaps are closed here. Observations are re-keyed to ``agent_pos`` /
    ``environment_state`` / ``pixels``, which is what
    ``lerobot.envs.utils.preprocess_observation`` reads. ``info["success"]`` is
    mirrored to ``info["is_success"]``, the key LeRobot's rollout pulls out of
    ``info["final_info"]``. And ``task_description`` is re-exposed, because
    Gymnasium 1.x wrappers no longer forward attributes and LeRobot reads that
    one straight off the sub-environment.

    Parameters
    ----------
    env
        A constructed SO101-Nexus MuJoCo environment.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self._joint_slice = _joint_positions_slice(cast("Any", env.unwrapped).config.observations)
        self.observation_space = _adapt_space(env.observation_space, self._joint_slice)

    @property
    def task_description(self) -> str:
        """Language instruction for the current episode."""
        return cast("Any", self.env.unwrapped).task_description

    @property
    def task(self) -> str:
        """Alias of ``task_description``: LeRobot probes for both names."""
        return self.task_description

    def reset(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset the environment and return a re-keyed observation."""
        observation, info = self.env.reset(**kwargs)
        return _adapt_observation(observation, self._joint_slice), _with_is_success(info)

    def step(self, action: Any) -> tuple[dict[str, Any], Any, Any, Any, dict[str, Any]]:
        """Step the environment and return a re-keyed observation."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        return (
            _adapt_observation(observation, self._joint_slice),
            reward,
            terminated,
            truncated,
            _with_is_success(info),
        )


def _with_is_success(info: dict[str, Any]) -> dict[str, Any]:
    """Mirror ``success`` to the ``is_success`` key LeRobot's rollout reads."""
    info["is_success"] = bool(info.get("success", False))
    return info


def _to_numpy(value: Any) -> np.ndarray:
    """Return ``value`` as a NumPy array, detaching a torch tensor if needed."""
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


class _WarpWorld:
    """Per-world view of a batched Warp env, for readers of ``VectorEnv.envs``."""

    __slots__ = ("_env", "_index")

    def __init__(self, env: Any, index: int) -> None:
        self._env = env
        self._index = index

    @property
    def task_description(self) -> str:
        """Language instruction for this world's current episode."""
        return self._env.task_descriptions[self._index]

    @property
    def task(self) -> str:
        """Alias of ``task_description``: LeRobot probes for both names."""
        return self.task_description


class WarpEnvHubAdapter(gym.vector.VectorWrapper):
    """Present a batched Warp environment as the NumPy vector env LeRobot expects.

    The Warp backend steps every world in one process on one device and speaks
    torch tensors. LeRobot's rollout speaks NumPy, indexes ``VectorEnv.envs`` for
    the per-world task string, and reads success out of ``info["final_info"]``,
    so this adapter converts at the boundary. The conversion copies device
    tensors to host memory every step, which is the cost of the NumPy contract;
    train against the Warp env directly (``gymnasium.make_vec``) to keep the
    batch on the GPU.

    Same-step autoreset is the Warp backend's own contract, and it does not keep
    the pre-reset observation, so ``info["final_obs"]`` is not provided.

    Parameters
    ----------
    env
        A constructed Warp vector environment.
    """

    def __init__(self, env: gym.vector.VectorEnv) -> None:
        super().__init__(env)
        inner = cast("Any", env.unwrapped)
        self._device = inner.device
        self._joint_slice = _joint_positions_slice(inner.config.observations)
        self.single_observation_space = _adapt_space(
            env.single_observation_space, self._joint_slice
        )
        self.observation_space = batch_space(self.single_observation_space, env.num_envs)
        self.envs = [_WarpWorld(inner, index) for index in range(env.num_envs)]

    def reset(
        self,
        *,
        seed: int | list[int] | tuple[int, ...] | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset every world. A per-world seed sequence collapses to its first entry.

        The Warp backend draws every world's episode from one seeded generator,
        so it takes a single seed rather than one per world.
        """
        if isinstance(seed, list | tuple):
            seed = int(seed[0]) if seed else None
        observation, info = self.env.reset(seed=seed, options=options)
        return self._observation(observation), _numpy_info(info)

    def step(
        self, actions: np.ndarray
    ) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        """Step every world from a NumPy action batch."""
        import torch

        tensor = torch.as_tensor(np.asarray(actions), dtype=torch.float32, device=self._device)
        observation, reward, terminated, truncated, info = self.env.step(tensor)
        terminated = _to_numpy(terminated)
        truncated = _to_numpy(truncated)
        info = _numpy_info(info)
        done = terminated | truncated
        if done.any():
            success = info.get("success")
            reported = np.zeros_like(done) if success is None else np.asarray(success, bool)
            info["final_info"] = {"is_success": reported & done}
        return self._observation(observation), _to_numpy(reward), terminated, truncated, info

    def call(self, name: str, *args: Any, **kwargs: Any) -> tuple[Any, ...]:
        """Return one value per world for the attribute or method ``name``."""
        inner = self.env.unwrapped
        if name in ("task_description", "task"):
            return tuple(inner.task_descriptions)
        if name == "_max_episode_steps":
            return (inner.max_episode_steps,) * self.num_envs
        attribute = getattr(inner, name)
        value = attribute(*args, **kwargs) if callable(attribute) else attribute
        # A per-world value is already one entry per world; a scalar describes the
        # whole batch, so it is what every world reports.
        if not isinstance(value, str) and hasattr(value, "__len__") and len(value) == self.num_envs:
            return tuple(value)
        return (value,) * self.num_envs

    def _observation(self, observation: Any) -> dict[str, Any]:
        """Convert one batched torch observation into NumPy LeRobot keys."""
        if isinstance(observation, dict):
            observation = {key: _to_numpy(value) for key, value in observation.items()}
        else:
            observation = _to_numpy(observation)
        return _adapt_observation(observation, self._joint_slice)


def _numpy_info(info: dict[str, Any]) -> dict[str, Any]:
    """Convert a batched Warp ``info`` dict's tensor values to NumPy arrays."""
    return {
        key: _to_numpy(value) if hasattr(value, "detach") else value for key, value in info.items()
    }
