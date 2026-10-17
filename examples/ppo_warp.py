"""PPO (CleanRL style) for the GPU-batched SO101-Nexus Warp envs.

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

The training env uses the all-observation default config (31-d privileged state:
`joints(6) + joint_vels(6) + ee_pose(7) + grasp(1) + gaze(1) + obj_pose(7) + obj_offset(3)`),
so no custom observation list is needed. `best_agent.pt` (model + obs-norm stats) is saved on
success-rate improvement.

Usage::

    uv run --extra warp --extra train python examples/ppo_warp.py
    uv run --extra warp --extra train python examples/ppo_warp.py --num-envs 512 --seed 3
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from so101_nexus import privileged_state_feature_names
from so101_nexus._reproducibility import seed_everything
from so101_nexus._run_metadata import training_metadata

try:
    from examples import _warp_training as _training
except ImportError:  # Direct script execution adds examples/ to sys.path.
    import _warp_training as _training  # ty: ignore[unresolved-import]


@dataclass
class Args:
    """PPO hyperparameters and run configuration (CleanRL layout)."""

    exp_name: str = "ppo"
    seed: int = 1
    torch_deterministic: bool = True
    torch_deterministic_warn_only: bool = False
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


_finite = _training._finite
RunningMeanStd = _training.RunningMeanStd
ObsNormalizer = _training.ObsNormalizer
RewardScaler = _training.RewardScaler
layer_init = _training.layer_init
Agent = _training.Agent
_env_cls_from_spec = _training._env_cls_from_spec
_resolve_env_cls = _training._resolve_env_cls
_mujoco_config = _training._mujoco_config
_make_envs = _training._make_envs
evaluate_mujoco = _training.evaluate_mujoco
write_video = _training.write_video


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
    return _training.rollout_video_from_checkpoint(
        checkpoint,
        env_id,
        control_mode=control_mode,
        episode_length=episode_length,
        hidden_dim=hidden_dim,
        seed=seed,
        out_path=out_path,
        fps=fps,
        capture_video=capture_video,
        evaluate=evaluate_mujoco,
        video_writer=write_video,
        normalizer_cls=ObsNormalizer,
        metadata_layout=True,
    )


_save = _training._save


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
    torch_deterministic_warn_only=False,
    save_dir=None,
    writer=None,
    log_freq=1,
    log=False,
):
    """Run PPO on the batched Warp env; return summary stats. Stays on ``device``.

    Batch-size arguments are validated before any environment is constructed, so
    invalid configurations fail fast without paying the Warp model-build cost.
    """
    run_config = locals().copy()
    run_config.pop("writer")
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

    seed_everything(
        seed,
        deterministic=torch_deterministic,
        deterministic_warn_only=torch_deterministic_warn_only,
    )
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
    env_state_names = privileged_state_feature_names(envs.unwrapped.config.observations)
    run_metadata = training_metadata(
        device=dev,
        env_config=envs.unwrapped.config,
        env_id=env_id,
        control_mode=control_mode,
        episode_length=episode_length,
        hidden_dim=hidden_dim,
        seed=seed,
        env_state_names=env_state_names,
        run_config=run_config,
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

    episode_stats = torch.empty((num_steps, num_envs, 4), device=dev)
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
    clipfracs: list[torch.Tensor] = []
    succ_rate = mean_ret = mean_len = 0.0

    for update in range(1, num_updates + 1):
        if anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            optimizer.param_groups[0]["lr"] = frac * learning_rate
        ent_now = ent_coef + (ent_coef_final - ent_coef) * ((update - 1.0) / num_updates)

        hold_sum = torch.zeros((), device=dev)
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
            hold_sum += succ.float().mean()

            # Fire success_bonus once, on the step success is first reached: fixed horizon
            # suppresses `terminated`, so key off the true info["success"] instead.
            first_success = succ & ~ep_succeeded
            shaped = reward + success_bonus * first_success.float()
            rew_buf[step] = rew_scaler(shaped, done)

            # episodic bookkeeping on the RAW env reward / true success
            ep_ret += reward
            ep_len += 1
            ep_succeeded |= succ
            episode_stats[step] = torch.stack((ep_ret, ep_len, ep_succeeded, done), dim=-1)
            done_mask = done.bool()
            ep_ret.masked_fill_(done_mask, 0.0)
            ep_len.masked_fill_(done_mask, 0.0)
            ep_succeeded &= ~done_mask

            next_obs = obs_norm(next_obs_raw, update=True)
            next_done = done

        # Transfer episode diagnostics once per rollout, preserving step/world order.
        completed = episode_stats.cpu()
        for r_, l_, s_ in completed[completed[..., 3].bool(), :3].tolist():
            ret_hist.append(r_)
            len_hist.append(l_)
            succ_hist.append(s_)

        with torch.no_grad():
            next_value = agent.get_value(next_obs).squeeze(-1)
            advantages, returns = _training.generalized_advantage_estimate(
                rew_buf, val_buf, done_buf, next_value, next_done, gamma, gae_lambda
            )

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
                pg_loss, v_loss, approx_kl, clipfrac = _training.ppo_losses(
                    newlogp,
                    b_logp[mb],
                    newval,
                    b_val[mb],
                    b_adv[mb],
                    b_ret[mb],
                    clip_coef,
                    norm_adv,
                    clip_vloss,
                )
                clipfracs.append(clipfrac)

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
        hold_frac = float(hold_sum / num_steps)
        sps = int(global_step / (time.time() - start_time))

        if succ_rate > best_success and len(succ_hist) >= 100:
            best_success = succ_rate
            if save_dir:
                _save(
                    agent,
                    obs_norm,
                    f"{save_dir}/best_agent.pt",
                    global_step,
                    succ_rate,
                    metadata=run_metadata,
                    env_config=envs.unwrapped.config,
                )

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
                "losses/clipfrac": float(torch.stack(clipfracs).mean()) if clipfracs else 0.0,
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
        _save(
            agent,
            obs_norm,
            f"{save_dir}/agent.pt",
            global_step,
            succ_rate,
            metadata=run_metadata,
            env_config=envs.unwrapped.config,
        )

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
        torch_deterministic_warn_only=args.torch_deterministic_warn_only,
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
