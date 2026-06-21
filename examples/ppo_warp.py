"""Train PPO on the GPU-batched WarpTouch environment, entirely on-device.

Unlike ``examples/ppo.py`` (CPU MuJoCo + SyncVectorEnv with numpy round-trips),
this consumes the native batched Warp env whose reset/step return torch tensors
on the GPU, so the rollout never leaves the device.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn, optim
from torch.distributions import Normal

_LAYER_INIT_STD = float(np.sqrt(2))


def layer_init(layer, std=_LAYER_INIT_STD, bias_const=0.0):
    """Orthogonal-initialize a linear layer."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """MLP actor-critic (mirrors examples/ppo.py)."""

    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
        )
        self.actor_mean = layer_init(nn.Linear(256, act_dim), std=0.01)
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))
        self.critic = layer_init(nn.Linear(256, 1), std=1.0)

    def get_value(self, x):
        return self.critic(self.network(x))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x)
        mean = self.actor_mean(hidden)
        std = self.actor_logstd.expand_as(mean).exp()
        probs = Normal(mean, std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(hidden)


def _make_envs(env_id, num_envs, device, seed):
    import gymnasium as gym

    import so101_nexus.warp  # noqa: F401
    from so101_nexus.config import TouchConfig
    from so101_nexus.observations import JointPositions, ObjectOffset

    # ObjectOffset makes the target object observable so the policy can learn
    # (the touch task is object-centric; add it explicitly for a compact obs).
    config = TouchConfig(observations=[JointPositions(), ObjectOffset()])
    return gym.make_vec(
        env_id,
        num_envs=num_envs,
        config=config,
        device=device,
        seed=seed,
        vectorization_mode="vector_entry_point",
    )


# Cohesive single-file CleanRL-style PPO training loop; kept as one function.
def train(  # noqa: PLR0915, PLR0912, C901
    *,
    env_id="WarpTouch-v1",
    num_envs=4096,
    num_steps=16,
    total_timesteps=5_000_000,
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    num_minibatches=32,
    update_epochs=4,
    clip_coef=0.2,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
    norm_adv=True,
    device="cuda",
    seed=0,
    log=False,
):
    """Run PPO on the batched Warp env; return summary stats. Stays on `device`."""
    batch_size = num_envs * num_steps
    if num_minibatches < 1 or num_minibatches > batch_size:
        raise ValueError(f"num_minibatches must be in [1, {batch_size}], got {num_minibatches}")
    if batch_size % num_minibatches != 0:
        raise ValueError(
            f"batch_size ({batch_size}) must be divisible by num_minibatches ({num_minibatches})"
        )
    if total_timesteps < batch_size:
        raise ValueError(
            f"total_timesteps ({total_timesteps}) must be >= batch_size "
            f"({batch_size}) for at least one training iteration"
        )
    minibatch_size = batch_size // num_minibatches
    num_iterations = total_timesteps // batch_size

    torch.manual_seed(seed)
    dev = torch.device(device)
    envs = _make_envs(env_id, num_envs, device, seed)
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    act_dim = int(np.prod(envs.single_action_space.shape))

    agent = Agent(obs_dim, act_dim).to(dev)
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)

    obs = torch.zeros((num_steps, num_envs, obs_dim), device=dev)
    actions = torch.zeros((num_steps, num_envs, act_dim), device=dev)
    logprobs = torch.zeros((num_steps, num_envs), device=dev)
    rewards = torch.zeros((num_steps, num_envs), device=dev)
    dones = torch.zeros((num_steps, num_envs), device=dev)
    values = torch.zeros((num_steps, num_envs), device=dev)

    next_obs, _ = envs.reset(seed=seed)
    next_obs = next_obs.to(dev)
    next_done = torch.zeros(num_envs, device=dev)
    ep_return = torch.zeros(num_envs, device=dev)
    returns_log: list[float] = []
    start = time.time()
    global_step = 0
    pg_loss = v_loss = torch.tensor(0.0)

    for _it in range(num_iterations):
        for step in range(num_steps):
            global_step += num_envs
            obs[step] = next_obs
            dones[step] = next_done
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminated, truncated, _ = envs.step(action)
            next_obs = next_obs.to(dev)
            reward = reward.to(dev)
            done = (terminated | truncated).to(dtype=torch.float32, device=dev)
            rewards[step] = reward
            ep_return += reward
            finished = done.bool()
            if bool(finished.any()):
                returns_log.extend(ep_return[finished].tolist())
                ep_return[finished] = 0.0
            next_done = done

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = (
                    delta + gamma * gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        b_obs = obs.reshape((-1, obs_dim))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1, act_dim))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        for _epoch in range(update_epochs):
            b_inds = torch.randperm(batch_size, device=dev)
            for s in range(0, batch_size, minibatch_size):
                mb = b_inds[s : s + minibatch_size]
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb], b_actions[mb]
                )
                ratio = (newlogprob - b_logprobs[mb]).exp()
                mb_adv = b_advantages[mb]
                if norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                pg_loss = torch.max(
                    -mb_adv * ratio,
                    -mb_adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef),
                ).mean()
                v_loss = 0.5 * ((newvalue.view(-1) - b_returns[mb]) ** 2).mean()
                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + vf_coef * v_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()

        if log:
            mean_ret = float(np.mean(returns_log[-100:])) if returns_log else float("nan")
            sps = int(global_step / (time.time() - start))
            print(f"step={global_step} mean_return={mean_ret:.3f} SPS={sps}")

    envs.close()
    return {
        "iterations": num_iterations,
        "policy_loss": float(pg_loss.item()),
        "value_loss": float(v_loss.item()),
        "mean_return": float(np.mean(returns_log[-100:])) if returns_log else float("nan"),
        "episodes": len(returns_log),
    }


@dataclass
class Args:
    env_id: str = "WarpTouch-v1"
    num_envs: int = 4096
    num_steps: int = 16
    total_timesteps: int = 5_000_000
    learning_rate: float = 3e-4
    num_minibatches: int = 32
    update_epochs: int = 4
    device: str = "cuda"
    seed: int = 0


if __name__ == "__main__":
    import tyro

    args = tyro.cli(Args)
    stats = train(
        env_id=args.env_id,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        num_minibatches=args.num_minibatches,
        update_epochs=args.update_epochs,
        device=args.device,
        seed=args.seed,
        log=True,
    )
    print(stats)
