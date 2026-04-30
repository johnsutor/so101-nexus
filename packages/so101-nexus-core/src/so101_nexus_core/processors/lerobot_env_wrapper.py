"""Gym wrapper that emits LeRobot ``EnvTransition`` shape observations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import gymnasium as gym
import numpy as np

from so101_nexus_core.processors.pipelines import (
    _infer_rename_map,
    make_default_env_observation_pipeline,
)

if TYPE_CHECKING:
    import torch
    from lerobot.processor import DataProcessorPipeline


def _derive_observation_space(
    base_space: gym.spaces.Dict,
    rename_map: dict[str, str],
    image_keys: tuple[str, ...],
    add_batch_dim: bool,
) -> gym.spaces.Dict:
    """Build a ``gym.spaces.Dict`` reflecting the post-pipeline observation shape."""
    new_spaces: dict[str, gym.Space] = {}
    for key, space in base_space.spaces.items():
        new_key = rename_map.get(key, key)
        if new_key in image_keys and isinstance(space, gym.spaces.Box):
            h, w, c = space.shape
            shape = (1, c, h, w) if add_batch_dim else (c, h, w)
            new_spaces[new_key] = gym.spaces.Box(low=0.0, high=1.0, shape=shape, dtype=np.float32)
        elif isinstance(space, gym.spaces.Box) and add_batch_dim:
            new_spaces[new_key] = gym.spaces.Box(
                low=space.low[None, ...],
                high=space.high[None, ...],
                shape=(1, *space.shape),
                dtype=np.dtype(space.dtype).type,
            )
        else:
            new_spaces[new_key] = space
    return gym.spaces.Dict(new_spaces)


class LeRobotEnvWrapper(gym.ObservationWrapper):
    """Wrap a so101-nexus env so its observations match LeRobot conventions.

    The wrapper runs a configurable :class:`lerobot.processor.DataProcessorPipeline`
    on every observation returned by ``reset`` and ``step``. When ``pipeline`` is
    ``None``, the default pipeline (rename keys, HWC->CHW float32, optional batch
    and device) is built from the wrapped env's ``observation_space``.

    Parameters
    ----------
    env
        Underlying gym environment whose ``observation_space`` is a
        :class:`gym.spaces.Dict`.
    pipeline
        Optional override pipeline. When ``None`` a sensible default is built.
    device
        If set, the default pipeline appends a ``DeviceProcessorStep`` to move
        tensors onto this device.
    add_batch_dim
        If true, the default pipeline appends an ``AddBatchDimensionProcessorStep``.
    """

    def __init__(
        self,
        env: gym.Env,
        pipeline: DataProcessorPipeline | None = None,
        *,
        device: str | torch.device | None = None,
        add_batch_dim: bool = False,
    ) -> None:
        super().__init__(env)
        if not isinstance(env.observation_space, gym.spaces.Dict):
            raise TypeError(
                "LeRobotEnvWrapper requires a Dict observation_space; "
                f"got {type(env.observation_space).__name__}.",
            )
        self.pipeline = pipeline or make_default_env_observation_pipeline(
            env.observation_space,
            device=device,
            add_batch_dim=add_batch_dim,
        )
        rename_map = _infer_rename_map(env.observation_space.spaces.keys())
        image_keys = tuple(v for k, v in rename_map.items() if k.endswith("_camera"))
        self.observation_space = _derive_observation_space(
            env.observation_space,
            rename_map=rename_map,
            image_keys=image_keys,
            add_batch_dim=add_batch_dim,
        )

    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Run the configured pipeline over the observation dict."""
        return self.pipeline({"observation": observation})


def make_lerobot_env(
    env_id: str,
    *,
    pipeline: DataProcessorPipeline | None = None,
    device: str | torch.device | None = None,
    add_batch_dim: bool = False,
    **make_kwargs: Any,
) -> gym.Env:
    """Build a :class:`LeRobotEnvWrapper` around a registered gym env id."""
    base = gym.make(env_id, **make_kwargs)
    return LeRobotEnvWrapper(
        base, pipeline=pipeline, device=device, add_batch_dim=add_batch_dim,
    )
