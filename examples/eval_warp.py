"""Deterministic evaluation of a `ppo_warp.py` / `bc_ppo_warp.py` checkpoint on Warp.

Loads a saved agent + obs-normalization stats, runs the *deterministic* policy
(mean action, no exploration noise) for full fixed-horizon episodes across a batch of
Warp worlds, and reports the success rate (fraction of episodes that solve the task at
any point), the hold rate at the final step, and mean return. This is the number
quoted in the README (``success_rate(ever) = 0.998`` for the reference `WarpPickLift-v1`
checkpoint).

Usage::

    uv run --extra warp python examples/eval_warp.py --checkpoint runs/.../best_agent.pt
    uv run --extra warp python examples/eval_warp.py  # globs the latest best_agent.pt

When a checkpoint records the observation layout it was trained on, the env is rebuilt
with that layout so the saved network's input width still matches. ``bc_ppo_warp.py``
records the layout its demo dataset declares, which is narrower than the env default
whenever a component joined the defaults after the demos were recorded.
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import glob
from dataclasses import dataclass

import numpy as np
import torch

from so101_nexus import observations_from_feature_names, privileged_state_feature_names

try:
    from examples.ppo_warp import Agent, _make_envs, _resolve_env_cls
except ImportError:  # when run as `python examples/eval_warp.py`
    from ppo_warp import (  # ty: ignore[unresolved-import]
        Agent,
        _make_envs,
        _resolve_env_cls,
    )


@dataclass
class Args:
    """Deterministic-eval configuration."""

    checkpoint: str = "runs/WarpPickLift-v1__*/best_agent.pt"
    """path or glob to a training checkpoint (latest match is used for a glob)"""
    env_id: str = "WarpPickLift-v1"
    num_envs: int = 512
    episode_length: int = 512
    seed: int = 12345
    control_mode: str = "pd_joint_delta_pos"
    hidden_dim: int = 256


def main():
    """Run the deterministic policy across a batch of worlds and print metrics."""
    import tyro

    args = tyro.cli(Args)
    matches = sorted(glob.glob(args.checkpoint)) if "*" in args.checkpoint else [args.checkpoint]
    if not matches:
        raise FileNotFoundError(f"no checkpoint matched {args.checkpoint!r}")
    path = matches[-1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    print(f"[eval] checkpoint={path} trained_step={ckpt['step']} saved_success={ckpt['success']}")

    names = ckpt.get("env_state_names")
    default = privileged_state_feature_names(
        _resolve_env_cls(args.env_id).default_config_cls().observations
    )
    if names and list(names) != default:
        # Silently evaluating a stale layout would report plausible-looking numbers.
        print(
            f"[eval] checkpoint pins a {len(names)}-dim layout, not the {len(default)}-dim default"
        )
    envs = _make_envs(
        args.env_id,
        args.num_envs,
        device,
        args.seed,
        control_mode=args.control_mode,
        episode_length=args.episode_length,
        observations=None if not names else observations_from_feature_names(names),
    )
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    act_dim = int(np.prod(envs.single_action_space.shape))
    agent = Agent(obs_dim, act_dim, args.hidden_dim).to(device)
    agent.load_state_dict(ckpt["model"])
    agent.eval()
    mean = ckpt["obs_mean"].to(device).float()
    var = ckpt["obs_var"].to(device).float()

    def norm(o):
        return ((o.to(device) - mean) / torch.sqrt(var + 1e-8)).clamp(-10.0, 10.0)

    obs, _ = envs.reset(seed=args.seed)
    ever = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
    last_succ = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
    ret = torch.zeros(args.num_envs, device=device)
    with torch.no_grad():
        for _ in range(args.episode_length):
            action = agent.actor_mean(norm(obs))  # deterministic mean action
            obs, reward, _term, _trunc, info = envs.step(action)
            ret += reward.to(device)
            last_succ = info["success"].to(device).bool()
            ever |= last_succ
    envs.close()

    print(
        f"[eval] episodes={args.num_envs} horizon={args.episode_length} "
        f"success_rate(ever)={ever.float().mean().item():.4f} "
        f"hold_rate(final)={last_succ.float().mean().item():.4f} "
        f"mean_return={ret.mean().item():.2f}"
    )


if __name__ == "__main__":
    main()
