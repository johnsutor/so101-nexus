"""PPO (CleanRL single-file style) for the GPU-batched SO101-Nexus Warp envs.

This is the "best PPO results" recipe for `so101-nexus`: large-batch PPO on the
GPU-parallel MuJoCo Warp backend. `mujoco_warp` steps thousands of worlds in place
and returns obs/reward/done as GPU tensors (zero host round-trip), so the whole hot
path stays on-device. On `WarpPickLift-v1` the default recipe now uses a strong
entropy warm-start with a nonzero floor plus CleanRL-style optimizer settings
because MuJoCo Warp contact physics is not bitwise deterministic on GPU. A 30M-step
sweep reached final success 0.97, 0.985, 0.965, and 0.97 on seeds 1, 2, 3, and 4.

Logging uses TensorBoard (`torch.utils.tensorboard.SummaryWriter`, very Colab
friendly): launch `tensorboard --logdir runs` (or `%tensorboard --logdir runs` in a
notebook) and watch `charts/success_rate` climb live. See
`examples/ppo_warp_colab.ipynb` for a self-contained Colab run.

Three findings are decisive (all baked into the defaults):

- **Fixed-horizon episodes** (`--no-terminate-on-success`, default). The env
  terminates on success; with a dense positive reward that makes *succeeding end the
  reward stream*, so vanilla PPO farms the reach reward forever and never lifts
  (return plateaus, 0% success). Fixed-horizon makes grasp+lift+**hold** (~1.0/step)
  beat hovering (~0.25/step).
- **CleanRL-style optimizer budget** (`--update-epochs 10 --num-minibatches 32`,
  `--max-grad-norm 0.5`, no target-KL stop). The larger update budget is slower per
  environment step, but it turns early grasp discoveries into stable lift policies.
- **Strong entropy warm-start with a nonzero floor** (`--ent-coef 0.03 --ent-coef-final 0.005`).
  The high initial entropy gives the policy enough exploration to discover grasps
  across seeds, and the nonzero floor keeps exploration alive so the policy can
  escape the reaching local optimum late in training. `--stagger-resets` (default)
  desynchronizes episode phases so discovery is less seed-fragile.

The training env uses the all-observation default config (24-d privileged state:
`joints(6) + ee_pose(7) + grasp(1) + obj_pose(7) + obj_offset(3)`), so no custom
observation list is needed. `best_agent.pt` (model + obs-norm stats) is saved on
success-rate improvement.

Usage::

    uv run --extra warp --extra train python examples/ppo_warp.py
    uv run --extra warp --extra train python examples/ppo_warp.py --num-envs 512 --seed 3
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import importlib
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from so101_nexus._reproducibility import seed_everything


@dataclass
class Args:
    """PPO hyperparameters and run configuration (CleanRL layout)."""

    exp_name: str = "ppo"
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True

    env_id: str = "WarpPickLift-v1"
    """any registered ``Warp*-v1`` id; the recipe is tuned for ``WarpPickLift-v1``"""
    num_envs: int = 1024
    """number of GPU-parallel Warp worlds"""
    episode_length: int = 512
    """max steps per episode (truncation horizon)"""
    control_mode: str = "pd_joint_delta_pos"
    terminate_on_success: bool = False
    """default False = fixed-horizon (avoids the reach-farming trap; see module doc)"""
    success_bonus: float = 0.0
    """optional extra reward added on the step success is achieved (raw, pre-scale)"""
    stagger_resets: bool = True
    """randomize initial episode phase per env so the worlds do not reset in lockstep"""

    total_timesteps: int = 30_000_000
    learning_rate: float = 3e-4
    anneal_lr: bool = True
    num_steps: int = 16
    """rollout length per env per update; batch = num_envs * num_steps"""
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 32
    update_epochs: int = 10
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.03
    """entropy bonus; strong warm-start plus nonzero floor for consistent lift discovery"""
    ent_coef_final: float = 0.005
    """entropy coef is linearly annealed ent_coef -> ent_coef_final over training"""
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None
    """KL early-stop threshold (None disables, matching CleanRL continuous PPO)"""

    norm_obs: bool = True
    norm_reward: bool = True
    hidden_dim: int = 256

    track: bool = True
    """log scalars to TensorBoard under runs/{run_name}"""
    save_model: bool = True
    log_freq: int = 1
    capture_video: bool = True
    """render periodic deterministic eval rollouts in the matching MuJoCo backend"""
    eval_freq: int = 50
    """eval + video every N updates (rendered in the MuJoCo backend)"""
    eval_episodes: int = 5

    batch_size: int = field(default=0, init=False)
    minibatch_size: int = field(default=0, init=False)
    num_updates: int = field(default=0, init=False)


# One diverging Warp world in a batched rollout can emit a NaN/Inf observation
# or reward; left unguarded that value permanently poisons the shared running
# mean/variance accumulators below (every subsequent `update()` mixes NaN
# forward), eventually crashing training with e.g.
# `ValueError: Normal(loc=NaN, ...)`. `_finite` clamps such values to 0 / a
# large finite bound instead of letting them propagate.
_STAT_BOUND = 1e6


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
        return normed.clamp(-10.0, 10.0)


class RewardScaler:
    """Return-based reward scaling (gym NormalizeReward equivalent), on GPU."""

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
        return scaled


_LAYER_INIT_STD = float(np.sqrt(2))


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


def _resolve_env_cls(env_id: str):
    """Return the batched Warp env class registered under *env_id*."""
    import gymnasium as gym

    import so101_nexus.warp  # noqa: F401  registers Warp*-v1

    entry = gym.spec(env_id).vector_entry_point
    if not isinstance(entry, str):
        raise ValueError(f"{env_id} has no string vector_entry_point; not a Warp env")
    module_path, class_name = entry.split(":")
    return getattr(importlib.import_module(module_path), class_name)


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
):
    """Build the batched Warp training env with the all-observation default config."""
    env_cls = _resolve_env_cls(env_id)
    if not terminate_on_success:
        env_cls = _fixed_horizon(env_cls)
    return env_cls(
        num_envs=num_envs,
        config=None,  # env builds its all-observation default (privileged state)
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
):
    """Deterministic eval in the matching ``MuJoCo*`` backend (a transfer figure).

    Warp's render() is a no-op, so eval rollouts are rendered in the MuJoCo backend
    with the same default (all-observation) config and control mode; the saved Warp
    policy + obs-norm stats transfer directly (slight physics gap: Warp uses
    implicit/no-noslip). Returns (eval_metrics, frames) where frames is one episode's
    HxWx3 uint8 arrays (None when capture_video is False).
    """
    import gymnasium as gym

    import so101_nexus.mujoco  # noqa: F401 registers MuJoCo* envs

    env = gym.make(
        env_id.replace("Warp", "MuJoCo"),
        control_mode=control_mode,
        render_mode="rgb_array" if capture_video else None,
        max_episode_steps=episode_length,
    )
    mean = obs_norm.rms.mean.to(device).float()
    var = obs_norm.rms.var.to(device).float()

    def norm(o):
        t = torch.as_tensor(o, dtype=torch.float32, device=device)
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
    control_mode: str = "pd_joint_delta_pos",
    episode_length: int = 512,
    hidden_dim: int = 256,
    seed: int = 12345,
    out_path: str = "runs/colab_rollout.mp4",
    fps: int = 30,
    capture_video: bool = True,
) -> tuple[dict[str, float], str | None]:
    """Render one deterministic MuJoCo rollout of a saved Warp PPO policy to mp4.

    The Warp backend steps thousands of worlds in parallel and does not render, so the
    rollout is shown in the matching ``MuJoCo*`` backend (same default all-observation
    config and control mode). It is a transfer figure: the saved policy and obs-norm
    stats transfer directly, with a slight physics gap versus Warp. Returns
    ``(metrics, video_path)`` where ``video_path`` is the written mp4, or ``None`` when
    ``capture_video`` is False.
    """
    import gymnasium as gym

    import so101_nexus.mujoco  # noqa: F401 registers MuJoCo* envs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)

    # Probe a throwaway MuJoCo env for obs/act dimensions (flat privileged state,
    # matching the Warp training obs). The checkpoint stores only the policy and
    # obs-norm stats, not the dims.
    probe = gym.make(
        env_id.replace("Warp", "MuJoCo"),
        control_mode=control_mode,
        max_episode_steps=episode_length,
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

    obs_norm = ObsNormalizer(obs_dim, device, enabled=True)
    obs_norm.rms.mean = ckpt["obs_mean"].to(device).double()
    obs_norm.rms.var = ckpt["obs_var"].to(device).double()

    metrics, frames = evaluate_mujoco(
        agent,
        obs_norm,
        device,
        env_id=env_id,
        control_mode=control_mode,
        episode_length=episode_length,
        eval_episodes=1,
        seed=seed,
        capture_video=capture_video,
    )
    video_path = write_video(frames, out_path, fps=fps) if capture_video else None
    return metrics, video_path


def _save(agent, obs_norm, path, step, success):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model": agent.state_dict(),
            "obs_mean": obs_norm.rms.mean.cpu(),
            "obs_var": obs_norm.rms.var.cpu(),
            "step": step,
            "success": success,
        },
        path,
    )


def train(  # noqa: PLR0915, PLR0912, C901
    *,
    env_id="WarpPickLift-v1",
    num_envs=1024,
    num_steps=16,
    total_timesteps=30_000_000,
    learning_rate=3e-4,
    anneal_lr=True,
    gamma=0.99,
    gae_lambda=0.95,
    num_minibatches=32,
    update_epochs=10,
    norm_adv=True,
    clip_coef=0.2,
    clip_vloss=True,
    ent_coef=0.03,
    ent_coef_final=0.005,
    vf_coef=0.5,
    max_grad_norm=0.5,
    target_kl=None,
    norm_obs=True,
    norm_reward=True,
    hidden_dim=256,
    control_mode="pd_joint_delta_pos",
    episode_length=512,
    terminate_on_success=False,
    success_bonus=0.0,
    stagger_resets=True,
    capture_video=False,
    eval_freq=0,
    eval_episodes=5,
    device="cuda",
    seed=1,
    torch_deterministic=True,
    save_dir=None,
    writer=None,
    log_freq=1,
    log=False,
):
    """Run PPO on the batched Warp env; return summary stats. Stays on ``device``.

    Batch-size arguments are validated before any environment is constructed, so
    invalid configurations fail fast without paying the Warp model-build cost.
    """
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
    num_updates = total_timesteps // batch_size

    seed_everything(seed, deterministic=torch_deterministic)
    dev = torch.device(device)
    np_rng = np.random.default_rng(seed)
    policy_rng = torch.Generator(device=dev).manual_seed(seed + 1)
    stagger_rng = torch.Generator(device=dev).manual_seed(seed + 2)

    envs = _make_envs(
        env_id,
        num_envs,
        dev,
        seed,
        control_mode=control_mode,
        episode_length=episode_length,
        terminate_on_success=terminate_on_success,
    )
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    act_dim = int(np.prod(envs.single_action_space.shape))

    agent = Agent(obs_dim, act_dim, hidden_dim).to(dev)
    optimizer = torch.optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)
    obs_norm = ObsNormalizer(obs_dim, dev, enabled=norm_obs)
    rew_scaler = RewardScaler(num_envs, dev, gamma, enabled=norm_reward)

    obs_buf = torch.zeros((num_steps, num_envs, obs_dim), device=dev)
    act_buf = torch.zeros((num_steps, num_envs, act_dim), device=dev)
    logp_buf = torch.zeros((num_steps, num_envs), device=dev)
    rew_buf = torch.zeros((num_steps, num_envs), device=dev)
    done_buf = torch.zeros((num_steps, num_envs), device=dev)
    val_buf = torch.zeros((num_steps, num_envs), device=dev)

    ep_ret = torch.zeros(num_envs, device=dev)
    ep_len = torch.zeros(num_envs, device=dev)
    ep_succeeded = torch.zeros(num_envs, dtype=torch.bool, device=dev)
    ret_hist: deque = deque(maxlen=400)
    succ_hist: deque = deque(maxlen=400)
    len_hist: deque = deque(maxlen=400)

    global_step = 0
    start_time = time.time()
    best_success = -1.0

    next_obs_raw, _ = envs.reset(seed=seed)
    if stagger_resets:
        # Otherwise all worlds reset in lockstep (shared reset(seed)) and explore the
        # same episode phase together -> correlated, seed-fragile discovery. Random
        # initial elapsed staggers truncations so the batch spans all task phases.
        envs._elapsed = torch.randint(
            0, episode_length, (num_envs,), generator=stagger_rng, device=dev
        )
    next_obs = obs_norm(next_obs_raw.to(dev), update=True)
    next_done = torch.zeros(num_envs, device=dev)

    pg_loss = v_loss = entropy_loss = torch.tensor(0.0)
    approx_kl = torch.tensor(0.0)
    clipfracs: list[float] = []
    succ_rate = mean_ret = mean_len = 0.0

    for update in range(1, num_updates + 1):
        if anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            optimizer.param_groups[0]["lr"] = frac * learning_rate
        ent_now = ent_coef + (ent_coef_final - ent_coef) * ((update - 1.0) / num_updates)

        hold_sum = 0.0
        for step in range(num_steps):
            global_step += num_envs
            obs_buf[step] = next_obs
            done_buf[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(
                    next_obs, generator=policy_rng
                )
            act_buf[step] = action
            logp_buf[step] = logprob
            val_buf[step] = value

            next_obs_raw, reward, terminated, truncated, info = envs.step(action)
            next_obs_raw = _finite(next_obs_raw.to(dev))
            reward = _finite(reward.to(dev))
            terminated = terminated.to(dev)
            done = (terminated | truncated.to(dev)).float()
            succ = info["success"].to(dev).bool()
            hold_sum += float(succ.float().mean())

            # Fire success_bonus once, on the step success is first reached: fixed horizon
            # suppresses `terminated`, so key off the true info["success"] instead.
            first_success = succ & ~ep_succeeded
            shaped = reward + success_bonus * first_success.float()
            rew_buf[step] = rew_scaler(shaped, done)

            # episodic bookkeeping on the RAW env reward / true success
            ep_ret += reward
            ep_len += 1
            ep_succeeded |= succ
            done_mask = done.bool()
            if bool(done_mask.any()):
                idx = done_mask.nonzero(as_tuple=True)[0]
                for r_, l_, s_ in zip(
                    ep_ret[idx].tolist(),
                    ep_len[idx].tolist(),
                    ep_succeeded[idx].tolist(),
                    strict=False,
                ):
                    ret_hist.append(r_)
                    len_hist.append(l_)
                    succ_hist.append(float(s_))
                ep_ret[idx] = 0.0
                ep_len[idx] = 0.0
                ep_succeeded[idx] = False

            next_obs = obs_norm(next_obs_raw, update=True)
            next_done = done

        with torch.no_grad():
            next_value = agent.get_value(next_obs).squeeze(-1)
            advantages = torch.zeros_like(rew_buf)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - done_buf[t + 1]
                    nextvalues = val_buf[t + 1]
                delta = rew_buf[t] + gamma * nextvalues * nextnonterminal - val_buf[t]
                advantages[t] = lastgaelam = (
                    delta + gamma * gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + val_buf

        b_obs = obs_buf.reshape(-1, obs_dim)
        b_logp = logp_buf.reshape(-1)
        b_act = act_buf.reshape(-1, act_dim)
        b_adv = advantages.reshape(-1)
        b_ret = returns.reshape(-1)
        b_val = val_buf.reshape(-1)

        b_inds = np.arange(batch_size)
        clipfracs = []
        for _epoch in range(update_epochs):
            np_rng.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                mb = b_inds[start : start + minibatch_size]
                _, newlogp, entropy, newval = agent.get_action_and_value(b_obs[mb], b_act[mb])
                logratio = newlogp - b_logp[mb]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > clip_coef).float().mean().item())

                mb_adv = b_adv[mb]
                if norm_adv and mb_adv.numel() > 1:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                if clip_vloss:
                    v_unclipped = (newval - b_ret[mb]) ** 2
                    v_clipped = b_val[mb] + torch.clamp(newval - b_val[mb], -clip_coef, clip_coef)
                    v_clipped = (v_clipped - b_ret[mb]) ** 2
                    v_loss = 0.5 * torch.max(v_unclipped, v_clipped).mean()
                else:
                    v_loss = 0.5 * ((newval - b_ret[mb]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_now * entropy_loss + vf_coef * v_loss

                optimizer.zero_grad(set_to_none=True)
                # A NaN/Inf loss (from a diverging Warp world slipping past the
                # sanitization above) must not reach the optimizer: backward()
                # would NaN-poison every parameter's gradient, and step() would
                # then permanently corrupt the policy weights.
                if torch.isfinite(loss):
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                    optimizer.step()

            if target_kl is not None and approx_kl > target_kl:
                break

        succ_rate = float(np.mean(succ_hist)) if succ_hist else 0.0
        mean_ret = float(np.mean(ret_hist)) if ret_hist else 0.0
        mean_len = float(np.mean(len_hist)) if len_hist else 0.0
        hold_frac = hold_sum / num_steps
        sps = int(global_step / (time.time() - start_time))

        if succ_rate > best_success and len(succ_hist) >= 100:
            best_success = succ_rate
            if save_dir:
                _save(agent, obs_norm, f"{save_dir}/best_agent.pt", global_step, succ_rate)

        if (writer is not None or log) and (update % log_freq == 0 or update == num_updates):
            metrics = {
                "charts/success_rate": succ_rate,
                "charts/hold_frac": hold_frac,
                "charts/episodic_return": mean_ret,
                "charts/episodic_length": mean_len,
                "charts/SPS": sps,
                "charts/learning_rate": optimizer.param_groups[0]["lr"],
                "charts/entropy_coef": ent_now,
                "charts/best_success": max(best_success, 0.0),
                "losses/policy_loss": pg_loss.item(),
                "losses/value_loss": v_loss.item(),
                "losses/entropy": entropy_loss.item(),
                "losses/approx_kl": approx_kl.item(),
                "losses/clipfrac": float(np.mean(clipfracs)) if clipfracs else 0.0,
            }
            if writer is not None:
                for k, v in metrics.items():
                    writer.add_scalar(k, v, global_step)
            if log:
                print(
                    f"[step {global_step}] success={succ_rate:.3f} return={mean_ret:.1f} "
                    f"ent={ent_now:.4f} SPS={sps}",
                    flush=True,
                )

        if capture_video and eval_freq > 0 and (update % eval_freq == 0 or update == num_updates):
            try:
                ev, frames = evaluate_mujoco(
                    agent,
                    obs_norm,
                    dev,
                    env_id=env_id,
                    control_mode=control_mode,
                    episode_length=episode_length,
                    eval_episodes=eval_episodes,
                    seed=seed,
                    capture_video=True,
                )
            except Exception as e:  # eval is a best-effort transfer figure; never abort training
                print(f"[eval {global_step}] skipped: {e}", flush=True)
                ev, frames = {}, None
            if writer is not None:
                for k, v in ev.items():
                    writer.add_scalar(k, v, global_step)
            if log:
                print(f"[eval {global_step}] {ev}", flush=True)
            if frames and save_dir:
                write_video(frames, f"{save_dir}/videos/eval_{global_step}.mp4", fps=30)

    if save_dir:
        _save(agent, obs_norm, f"{save_dir}/agent.pt", global_step, succ_rate)

    envs.close()
    return {
        "iterations": num_updates,
        "policy_loss": float(pg_loss.item()),
        "value_loss": float(v_loss.item()),
        "mean_return": float(np.mean(ret_hist)) if ret_hist else float("nan"),
        "episodes": len(ret_hist),
        "success_rate": succ_rate,
        "best_success": max(best_success, 0.0),
    }


def main():
    """Parse CLI args, wire TensorBoard, and run the full training recipe."""
    import tyro

    args = tyro.cli(Args)
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_updates = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    save_dir = f"runs/{run_name}" if args.save_model else None

    writer = None
    if args.track:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n" + "\n".join(f"|{k}|{v}|" for k, v in vars(args).items()),
        )

    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if (torch.cuda.is_available() and args.cuda) else "cpu")
    print(f"[cfg] run={run_name} device={device} num_updates={args.num_updates}", flush=True)

    stats = train(
        env_id=args.env_id,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        anneal_lr=args.anneal_lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        num_minibatches=args.num_minibatches,
        update_epochs=args.update_epochs,
        norm_adv=args.norm_adv,
        clip_coef=args.clip_coef,
        clip_vloss=args.clip_vloss,
        ent_coef=args.ent_coef,
        ent_coef_final=args.ent_coef_final,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl,
        norm_obs=args.norm_obs,
        norm_reward=args.norm_reward,
        hidden_dim=args.hidden_dim,
        control_mode=args.control_mode,
        episode_length=args.episode_length,
        terminate_on_success=args.terminate_on_success,
        success_bonus=args.success_bonus,
        stagger_resets=args.stagger_resets,
        capture_video=args.capture_video,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        device=str(device),
        seed=args.seed,
        torch_deterministic=args.torch_deterministic,
        save_dir=save_dir,
        writer=writer,
        log_freq=args.log_freq,
        log=True,
    )

    print(
        f"[done] best_success={stats['best_success']:.3f} "
        f"final_success={stats['success_rate']:.3f}",
        flush=True,
    )
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
