#!/usr/bin/env python3
"""Smoke-check the installed so101-nexus package from PyPI."""

from __future__ import annotations

import argparse
import sys

import gymnasium as gym


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


def main() -> int:
    """Run the smoke check for the requested backend."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["mujoco"], default="mujoco")
    parser.add_argument("--env-id", required=True)
    args = parser.parse_args()

    print(f"Running smoke check for backend={args.backend}, env={args.env_id}")
    run_mujoco(args.env_id)
    print("Smoke check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
