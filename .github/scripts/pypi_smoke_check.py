#!/usr/bin/env python3
"""Smoke-check installed SO101-Nexus backend packages from PyPI."""

from __future__ import annotations

import argparse
import sys

import gymnasium as gym


def run_mujoco(env_id: str) -> None:
    import so101_nexus_mujoco  # noqa: F401

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


def run_maniskill(env_id: str) -> None:
    import so101_nexus_maniskill  # noqa: F401

    env = gym.make(env_id, obs_mode="state", num_envs=1, render_mode=None)
    try:
        obs, info = env.reset(seed=0)
        action = env.action_space.sample()
        step_result = env.step(action)
    finally:
        env.close()

    if obs is None or info is None:
        raise RuntimeError("ManiSkill smoke check failed: reset returned empty values.")
    if len(step_result) != 5:
        raise RuntimeError("ManiSkill smoke check failed: step did not return a 5-tuple.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["mujoco", "maniskill"], required=True)
    parser.add_argument("--env-id", required=True)
    args = parser.parse_args()

    print(f"Running smoke check for backend={args.backend}, env={args.env_id}")
    if args.backend == "mujoco":
        run_mujoco(args.env_id)
    else:
        run_maniskill(args.env_id)
    print("Smoke check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
