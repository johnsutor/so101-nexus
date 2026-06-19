#!/usr/bin/env python3
"""Manual MolmoAct2 rollout smoke test for SO101-Nexus environments."""

from __future__ import annotations

import argparse
import importlib
from typing import Any


def _dtype_from_name(name: str) -> Any | None:
    if name == "none":
        return None
    import torch

    return getattr(torch, name)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", default="MuJoCoPickLift-v1")
    parser.add_argument("--env-module", default=None, help="Optional module that registers env-id.")
    parser.add_argument("--repo-id", default="allenai/MolmoAct2-SO100_101")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=160)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device-map", default=None)
    parser.add_argument(
        "--dtype",
        choices=("none", "float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument(
        "--camera-keys",
        nargs="+",
        default=["overhead_camera", "wrist_camera"],
        help="Env camera keys in the order the policy expects.",
    )
    return parser.parse_args()


def _import_env_backend(env_id: str, env_module: str | None) -> None:
    if env_module:
        importlib.import_module(env_module)
        return

    from so101_nexus.teleop.leader import import_backend_for_env_id

    import_backend_for_env_id(env_id)


def main() -> None:
    """Run a real-model MolmoAct2 rollout smoke test."""
    args = _parse_args()
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
    main()
