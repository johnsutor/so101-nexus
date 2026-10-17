"""Shared numerical and rollout helpers for the Warp training recipes."""

from __future__ import annotations

import importlib
import os

import numpy as np
import torch
from torch import nn

from so101_nexus import observations_from_feature_names
from so101_nexus._reproducibility import seed_everything

_STAT_BOUND = 1e6
_LAYER_INIT_STD = float(np.sqrt(2))
_MISSING = object()


def _finite(x: torch.Tensor) -> torch.Tensor:
    """Replace non-finite entries with 0 (NaN) or a large finite bound (+-Inf)."""
    return torch.nan_to_num(x, nan=0.0, posinf=_STAT_BOUND, neginf=-_STAT_BOUND)


class RunningMeanStd:
    """On-GPU Welford running mean / variance for a fixed feature shape."""

    def __init__(self, shape, device, epsilon=1e-4):
        self.mean = torch.zeros(shape, dtype=torch.float64, device=device)
        self.var = torch.ones(shape, dtype=torch.float64, device=device)
        self.count = epsilon

    def update(self, x: torch.Tensor) -> None:
        x = _finite(x.to(torch.float64))
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        self.mean = self.mean + delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / tot
        self.var = m2 / tot
        self.count = tot


class ObsNormalizer:
    """Normalize observations with running stats; clip to +/-10 (CleanRL-style)."""

    sanitize_outputs = False

    def __init__(self, obs_dim, device, enabled=True):
        self.rms = RunningMeanStd((obs_dim,), device)
        self.enabled = enabled

    def __call__(self, obs: torch.Tensor, update: bool = True) -> torch.Tensor:
        if not self.enabled:
            return obs
        obs = _finite(obs)
        if update:
            self.rms.update(obs)
        normed = (obs - self.rms.mean.float()) / torch.sqrt(self.rms.var.float() + 1e-8)
        if self.sanitize_outputs:
            normed = torch.nan_to_num(normed, nan=0.0, posinf=10.0, neginf=-10.0)
        return normed.clamp(-10.0, 10.0)


class RewardScaler:
    """Return-based reward scaling (gym NormalizeReward equivalent), on GPU."""

    sanitize_outputs = False

    def __init__(self, num_envs, device, gamma, enabled=True):
        self.ret = torch.zeros(num_envs, dtype=torch.float64, device=device)
        self.rms = RunningMeanStd((), device)
        self.gamma = gamma
        self.enabled = enabled

    def __call__(self, reward: torch.Tensor, done: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return reward
        reward = _finite(reward)
        self.ret = self.ret * self.gamma + reward.to(torch.float64)
        self.rms.update(self.ret)
        scaled = reward / torch.sqrt(self.rms.var.float() + 1e-8)
        self.ret = self.ret * (1.0 - done.to(torch.float64))
        if self.sanitize_outputs:
            scaled = torch.nan_to_num(scaled, nan=0.0, posinf=1e3, neginf=-1e3)
        return scaled


def layer_init(layer, std=_LAYER_INIT_STD, bias_const=0.0):
    """Orthogonal-initialize a linear layer."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """MLP actor-critic with a state-independent log-std (clamped to std <= 1)."""

    def __init__(self, obs_dim, act_dim, hidden_dim):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, act_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(
        self,
        x,
        action=None,
        *,
        generator: torch.Generator | None = None,
    ):
        mean = self.actor_mean(x)
        logstd = self.actor_logstd.clamp(-5.0, 0.0).expand_as(mean)
        std = torch.exp(logstd)
        dist = torch.distributions.Normal(mean, std)
        if action is None:
            noise = torch.randn(
                mean.shape,
                dtype=mean.dtype,
                device=mean.device,
                generator=generator,
            )
            action = mean + std * noise
        logprob = dist.log_prob(action).sum(1)
        entropy = dist.entropy().sum(1)
        return action, logprob, entropy, self.critic(x).squeeze(-1)


def _env_cls_from_spec(env_id: str, attr: str):
    """Import the env class a Gymnasium spec's ``attr`` entry point names."""
    import gymnasium as gym

    import so101_nexus.mujoco
    import so101_nexus.warp  # noqa: F401  registers Warp*-v1

    entry = getattr(gym.spec(env_id), attr)
    if not isinstance(entry, str):
        raise ValueError(f"{env_id} has no string {attr}")
    module_path, class_name = entry.split(":")
    return getattr(importlib.import_module(module_path), class_name)


def _resolve_env_cls(env_id: str):
    """Return the batched Warp env class registered under *env_id*."""
    return _env_cls_from_spec(env_id, "vector_entry_point")


def _mujoco_config(mujoco_env_id: str, observations):
    """Build the MuJoCo env's config with a pinned observation layout."""
    return _env_cls_from_spec(mujoco_env_id, "entry_point").default_config_cls(
        observations=observations
    )


def _fixed_horizon(env_cls):
    """Subclass *env_cls* so episodes never terminate on success (fixed-length).

    Success is still reported via ``info['success']``; only the ``terminated`` flag
    is suppressed, so the reward stream continues and grasp+lift+hold beats hovering.
    """

    class _FixedHorizon(env_cls):
        def _compute_reward_terminated(self, *args, **kwargs):
            reward, success, info = super()._compute_reward_terminated(*args, **kwargs)
            return reward, torch.zeros_like(success), info

    _FixedHorizon.__name__ = f"FixedHorizon{env_cls.__name__}"
    return _FixedHorizon


def _make_envs(
    env_id,
    num_envs,
    device,
    seed,
    *,
    control_mode="pd_joint_delta_pos",
    episode_length=512,
    terminate_on_success=False,
    observations=None,
    config=None,
):
    """Build the batched Warp training env.

    ``observations=None`` takes the env's all-observation default config (the
    privileged state this script trains on); otherwise the layout is pinned, which
    is what evaluating a checkpoint trained against a different layout needs, such
    as a ``bc_ppo_warp.py`` run pinned to its demo dataset's recorded components.
    """
    env_cls = _resolve_env_cls(env_id)
    if config is None and observations is not None:
        config = env_cls.default_config_cls(observations=observations)
    if not terminate_on_success:
        env_cls = _fixed_horizon(env_cls)
    return env_cls(
        num_envs=num_envs,
        config=config,
        control_mode=control_mode,
        device=str(device),
        max_episode_steps=episode_length,
        seed=seed,
    )


def evaluate_mujoco(
    agent,
    obs_norm,
    device,
    *,
    env_id,
    control_mode,
    episode_length,
    eval_episodes,
    seed,
    capture_video,
    observations=None,
    config=None,
    sanitize_inputs=False,
):
    """Deterministic eval in the matching ``MuJoCo*`` backend (a transfer figure).

    Warp's render() is a no-op, so eval rollouts are rendered in the MuJoCo backend
    with the same config and control mode as training (``observations=None`` takes
    the all-observation default, otherwise the training layout is pinned); the saved
    Warp policy + obs-norm stats transfer directly (slight physics gap: Warp uses
    implicit/no-noslip). Returns (eval_metrics, frames) where frames is one episode's
    HxWx3 uint8 arrays (None when capture_video is False).
    """
    import gymnasium as gym

    import so101_nexus.mujoco  # noqa: F401 registers MuJoCo* envs

    mujoco_id = env_id.replace("Warp", "MuJoCo")
    env = gym.make(
        mujoco_id,
        control_mode=control_mode,
        render_mode="rgb_array" if capture_video else None,
        max_episode_steps=episode_length,
        **(
            {"config": config}
            if config is not None
            else (
                {} if observations is None else {"config": _mujoco_config(mujoco_id, observations)}
            )
        ),
    )
    mean = obs_norm.rms.mean.to(device).float()
    var = obs_norm.rms.var.to(device).float()

    def norm(o):
        t = torch.as_tensor(o, dtype=torch.float32, device=device)
        if sanitize_inputs:
            t = _finite(t)
        return (((t - mean) / torch.sqrt(var + 1e-8)).clamp(-10.0, 10.0)).unsqueeze(0)

    returns, succs, lens = [], [], []
    frames = None
    with torch.no_grad():
        for ep in range(eval_episodes):
            obs, _ = env.reset(seed=seed + 1000 + ep)
            ep_ret, ep_len, done = 0.0, 0, False
            ever_succ = False
            capture = capture_video and ep == 0
            if capture:
                frames = []
            while not done:
                a = agent.actor_mean(norm(obs)).squeeze(0).cpu().numpy()
                obs, r, term, trunc, info = env.step(a)
                ep_ret += float(r)
                ep_len += 1
                ever_succ = ever_succ or bool(info.get("success", False))
                done = bool(term or trunc)
                if capture and frames is not None:
                    frames.append(env.render())
            returns.append(ep_ret)
            succs.append(float(ever_succ))
            lens.append(ep_len)
    env.close()
    metrics = {
        "eval/return": float(np.mean(returns)),
        "eval/success_rate": float(np.mean(succs)),
        "eval/ep_len": float(np.mean(lens)),
    }
    return metrics, frames


def write_video(frames, path, fps=30):
    """Encode RGB frames to an mp4 (requires ``imageio[ffmpeg]``)."""
    try:
        import imageio.v2 as imageio
    except ImportError:
        print("[warn] imageio not installed; skipping video. `pip install 'imageio[ffmpeg]'`")
        return None
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, frames, fps=fps)
    return path


def rollout_video_from_checkpoint(
    checkpoint: str,
    env_id: str,
    *,
    control_mode: str | None = None,
    episode_length: int | None = None,
    hidden_dim: int | None = None,
    seed: int = 12345,
    out_path: str = "runs/colab_rollout.mp4",
    fps: int = 30,
    capture_video: bool = True,
    evaluate=evaluate_mujoco,
    video_writer=write_video,
    normalizer_cls=ObsNormalizer,
    metadata_layout=True,
) -> tuple[dict[str, float], str | None]:
    """Render one deterministic MuJoCo rollout of a saved Warp PPO policy to mp4.

    The Warp backend steps thousands of worlds in parallel and does not render, so the
    rollout is shown in the matching ``MuJoCo*`` backend (same control mode, and the
    observation layout the checkpoint records, falling back to the env default for a
    checkpoint written without it). It is a transfer figure: the saved policy and
    obs-norm stats transfer directly, with a slight physics gap versus Warp. Returns
    ``(metrics, video_path)`` where ``video_path`` is the written mp4, or ``None`` when
    ``capture_video`` is False.
    """
    import gymnasium as gym

    import so101_nexus.mujoco  # noqa: F401 registers MuJoCo* envs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(seed, deterministic=True)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    metadata = ckpt.get("metadata", {})
    control_mode = control_mode or metadata.get("control_mode", "pd_joint_delta_pos")
    episode_length = episode_length or metadata.get("episode_length", 512)
    hidden_dim = hidden_dim or metadata.get("hidden_dim", 256)

    names = ckpt.get("env_state_names")
    if metadata_layout and not names:
        names = metadata.get("env_state_names")
    observations = None if not names else observations_from_feature_names(names)
    saved_config = ckpt.get("env_config")

    # Probe a throwaway MuJoCo env for obs/act dimensions (flat privileged state,
    # matching the Warp training obs). The checkpoint stores the policy, obs-norm
    # stats, and the layout names, but not the dims.
    mujoco_id = env_id.replace("Warp", "MuJoCo")
    probe = gym.make(
        mujoco_id,
        control_mode=control_mode,
        max_episode_steps=episode_length,
        **(
            {"config": saved_config}
            if saved_config is not None
            else (
                {} if observations is None else {"config": _mujoco_config(mujoco_id, observations)}
            )
        ),
    )
    obs_shape = probe.observation_space.shape
    act_shape = probe.action_space.shape
    if obs_shape is None or act_shape is None:
        raise ValueError("rollout probe requires Box observation and action spaces")
    obs_dim = int(np.prod(obs_shape))
    act_dim = int(np.prod(act_shape))
    probe.close()

    agent = Agent(obs_dim, act_dim, hidden_dim).to(device)
    agent.load_state_dict(ckpt["model"])
    agent.eval()

    obs_norm = normalizer_cls(obs_dim, device, enabled=ckpt.get("norm_obs", True))
    obs_norm.rms.mean = ckpt["obs_mean"].to(device).double()
    obs_norm.rms.var = ckpt["obs_var"].to(device).double()

    metrics, frames = evaluate(
        agent,
        obs_norm,
        device,
        env_id=env_id,
        control_mode=control_mode,
        episode_length=episode_length,
        eval_episodes=1,
        seed=seed,
        capture_video=capture_video,
        observations=observations,
        config=saved_config,
    )
    video_path = video_writer(frames, out_path, fps=fps) if capture_video else None
    return metrics, video_path


def _save(
    agent,
    obs_norm,
    path,
    step,
    success,
    env_state_names=_MISSING,
    *,
    metadata=None,
    env_config=None,
):
    """Save an inference snapshot, not the state required to resume training."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            **({} if env_state_names is _MISSING else {"env_state_names": env_state_names}),
            "model": agent.state_dict(),
            "obs_mean": obs_norm.rms.mean.cpu(),
            "obs_var": obs_norm.rms.var.cpu(),
            "step": step,
            "success": success,
            "checkpoint_type": "inference",
            "checkpoint_version": 1,
            "norm_obs": obs_norm.enabled,
            "metadata": metadata or {},
            "env_config": env_config,
        },
        path,
    )


@torch.no_grad()
def generalized_advantage_estimate(
    rewards, values, dones, next_value, next_done, gamma, gae_lambda
):
    """Compute rollout advantages using the next observation's termination mask."""
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0
    for t in reversed(range(rewards.shape[0])):
        if t == rewards.shape[0] - 1:
            nextnonterminal = 1.0 - next_done
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - dones[t + 1]
            nextvalues = values[t + 1]
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
    return advantages, advantages + values


def ppo_losses(
    newlogp, oldlogp, newval, oldval, advantages, returns, clip_coef, norm_adv, clip_vloss
):
    """Compute clipped PPO losses and detached KL and clipping diagnostics."""
    logratio = newlogp - oldlogp
    ratio = logratio.exp()
    with torch.no_grad():
        approx_kl = ((ratio - 1) - logratio).mean()
        clipfrac = ((ratio - 1.0).abs() > clip_coef).float().mean()
    if norm_adv and advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
    if clip_vloss:
        v_unclipped = (newval - returns) ** 2
        v_clipped = oldval + torch.clamp(newval - oldval, -clip_coef, clip_coef)
        v_clipped = (v_clipped - returns) ** 2
        v_loss = 0.5 * torch.max(v_unclipped, v_clipped).mean()
    else:
        v_loss = 0.5 * ((newval - returns) ** 2).mean()
    return pg_loss, v_loss, approx_kl, clipfrac
