#!/usr/bin/env python3
"""Smoke-check the installed so101-nexus package from PyPI."""

from dataclasses import dataclass
from typing import Literal

import gymnasium as gym
import tyro


def run_mujoco(env_id: str) -> None:
    """Reset and step a registered MuJoCo env to verify the install works."""
    import so101_nexus.mujoco  # noqa: F401

    env = gym.make(env_id)
    try:
        obs, info = env.reset(seed=0)
        action = env.action_space.sample()
        step_result = env.step(action)
    finally:
        env.close()

    if obs is None or info is None:
        raise RuntimeError("MuJoCo smoke check failed: reset returned empty values.")
    if len(step_result) != 5:
        raise RuntimeError("MuJoCo smoke check failed: step did not return a 5-tuple.")


def run_warp(env_id: str, *, device: str, num_envs: int) -> None:
    """Reset and step a batched Warp vector env to verify the install works."""
    import so101_nexus.warp  # noqa: F401

    env = gym.make_vec(
        env_id, num_envs=num_envs, device=device, vectorization_mode="vector_entry_point"
    )
    try:
        obs, info = env.reset(seed=0)
        action = env.action_space.sample()
        step_result = env.step(action)
    finally:
        env.close()

    if obs is None or info is None:
        raise RuntimeError("Warp smoke check failed: reset returned empty values.")
    if len(step_result) != 5:
        raise RuntimeError("Warp smoke check failed: step did not return a 5-tuple.")


@dataclass
class Args:
    """Command-line arguments for the PyPI smoke check."""

    env_id: str
    backend: Literal["mujoco", "warp"] = "mujoco"
    device: str = "cpu"
    num_envs: int = 2


def main(args: Args) -> None:
    """Run the smoke check for the requested backend."""
    print(f"Running smoke check for backend={args.backend}, env={args.env_id}")
    if args.backend == "warp":
        run_warp(args.env_id, device=args.device, num_envs=args.num_envs)
    else:
        run_mujoco(args.env_id)
    print("Smoke check passed.")


if __name__ == "__main__":
    main(tyro.cli(Args))
