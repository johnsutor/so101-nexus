#!/usr/bin/env python3
"""Manual MolmoAct2 rollout smoke test for SO101-Nexus environments."""

import importlib
from dataclasses import dataclass, field
from typing import Literal

import tyro


@dataclass
class Args:
    """Command-line arguments for the MolmoAct2 rollout smoke test."""

    env_id: str = "MuJoCoPickLift-v1"
    env_module: str | None = None
    repo_id: str = "allenai/MolmoAct2-SO100_101"
    episodes: int = 1
    max_steps: int = 160
    seed: int = 0
    device_map: str | None = None
    dtype: Literal["none", "float32", "float16", "bfloat16"] = "bfloat16"
    chunk_size: int = 8
    camera_keys: list[str] = field(default_factory=lambda: ["overhead_camera", "wrist_camera"])


def _dtype_from_name(name: str):
    if name == "none":
        return None
    import torch

    return getattr(torch, name)


def _import_env_backend(env_id: str, env_module: str | None) -> None:
    if env_module:
        importlib.import_module(env_module)
        return

    from so101_nexus.teleop.leader import import_backend_for_env_id

    import_backend_for_env_id(env_id)


def main(args: Args) -> None:
    """Run a real-model MolmoAct2 rollout smoke test."""
    _import_env_backend(args.env_id, args.env_module)

    import gymnasium as gym

    from so101_nexus.policy_adapters import MolmoActPolicy, RolloutRecorder

    policy = MolmoActPolicy.from_pretrained(
        args.repo_id,
        device=args.device_map,
        dtype=_dtype_from_name(args.dtype),
        chunk_size=args.chunk_size,
    )
    env = gym.make(args.env_id, render_mode="rgb_array", control_mode="pd_joint_pos")
    try:
        recorder = RolloutRecorder(
            env,
            policy,
            camera_keys=tuple(args.camera_keys),
            max_steps_per_episode=args.max_steps,
        )
        results = recorder.record_episodes(args.episodes, seed=args.seed)
    finally:
        env.close()

    for episode_idx, result in enumerate(results):
        print(
            f"episode={episode_idx} steps={result.n_steps} "
            f"success={result.success} info={result.info}"
        )


if __name__ == "__main__":
    main(tyro.cli(Args))
