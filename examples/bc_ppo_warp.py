"""BC-seeded PPO (CleanRL style) for the GPU-batched Warp `WarpPickLift-v1` env.

Same large-batch PPO recipe as `ppo_warp.py` (fixed-horizon episodes, CleanRL optimizer
budget, entropy warm-start/anneal -- see that file's module doc for why each is decisive),
plus demonstration seeding: the actor is pretrained via behavior cloning (BC) on the 10
successful teleop episodes in
[`johnsutor/MuJoCoPickLift-v1`](https://huggingface.co/datasets/johnsutor/MuJoCoPickLift-v1) before
online PPO starts, and (optionally, `--bc-coef > 0`) a BC regularization term keeps pulling the
actor mean toward demo actions throughout online training.

This remains a separate recipe with its own defaults and BC-specific training phases.
Numerical operations, environment construction, and evaluation helpers are shared with
`ppo_warp.py` in `_warp_training.py` to keep fixes consistent across the recipes.

Why BC-seeding, not TD-MPC2's MPC-planning approach (see the CHANGELOG for that exploration
and its removal): `ppo_warp.py` already solves this task (~97-100% success in ~30 min on an
RTX 5090) purely through exploration, so the one thing worth fixing is its failure mode --
some seeds get stuck at a "grasp-hold-at-table" local optimum and never discover the lift,
purely from exploration luck. Seeding the actor near the demos' actual grasp-lift behavior
directly targets that, without touching PPO's own decisive fixes or trading away its GPU-batch
throughput for MPC planning's kernel-launch-bound cost.

**Action units.** The demo dataset records absolute joint-position targets (as commanded to a
`pd_joint_pos` teleop session), but this recipe drives the env in PPO's proven
`pd_joint_delta_pos` mode. Rather than switch control modes (risking the proven recipe), demo
actions are recomputed as the delta between consecutive recorded joint states
(`observation.state[t+1] - observation.state[t]`, the realized per-step motion), matching what
a delta controller would need to command to reproduce that trajectory, then normalized by the
same `_DELTA_ACTION_SCALE` the env applies internally.

**Also covers the harder `WarpPickAndPlace-v1` task, as opt-in CLI flags -- all off/at their
`WarpPickLift-v1`-safe default otherwise, so the recipe above is completely unaffected.**
`env_id`/`demo_repo` are already free-form (any `Warp*-v1` id / matching demo dataset works;
the training env and the MuJoCo evaluator are built with the observation layout the demo
dataset declares, so a dataset recorded before a component joined the env defaults still
trains a matching policy). Pick-and-place's `success` (object lowered back to
the goal *and* the whole arm simultaneously static -- a rarer two-condition event than
pick-lift's pure height-threshold success) needs two additional levers, both zero/off by
default:

- `--success-bonus 50` -- a one-time reward on first reaching `info['success']` in an episode,
  making the rare completion event's advantage swamp the surrounding dense reward once found.
- `--anneal-timesteps`/`--lr-min-frac` -- decouples `anneal_lr`/`ent_coef`'s annealing
  *horizon* from `total_timesteps` (`0`/`0.0` = anneal across the full `total_timesteps`,
  identical to the historical un-decoupled behavior). Lets `total_timesteps` run well past the
  point the schedule finishes annealing, holding LR/entropy at a floored value for the extended
  tail instead of stretching the schedule itself across the longer horizon (which reliably
  backfires once a task's convergence is slow enough that mid-training exploration noise still
  matters).

Validated recipe for `WarpPickAndPlace-v1` (3 seeds, `so101-nexus==0.4.10`)::

    uv run --extra warp --extra train python examples/bc_ppo_warp.py \\
        --env-id WarpPickAndPlace-v1 --demo-repo johnsutor/MuJoCoPickAndPlace-v1 \\
        --success-bonus 50 --total-timesteps 160000000 \\
        --anneal-timesteps 80000000 --lr-min-frac 0.1

reaches `best_success` `0.927` / `0.912` / `0.743` across seeds 1/2/3 (mean `0.861`, std
`0.083`); a 100-episode MuJoCo-transfer eval of the seed-1 checkpoint measured
`success_rate=0.870` (95% CI approx +/- 0.066). Dropping `--anneal-timesteps`/`--lr-min-frac`
(plain `--total-timesteps 80000000`) still works (mean `best_success=0.601`, std `0.134`, a
large win over this task's earlier ~9% ceiling under a since-fixed, exploitable reward) but is
both lower-mean and higher-variance across seeds than the decoupled-schedule extension above --
the extension is not a lucky-seed artifact, it wins the paired seed-for-seed comparison every
time (`+13.7`, `+40.7`, `+23.6` points).

Usage::

    uv run --extra warp --extra train python examples/bc_ppo_warp.py
    uv run --extra warp --extra train python examples/bc_ppo_warp.py --bc-coef 0.2 --seed 3
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import re
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from so101_nexus import observations_from_feature_names, privileged_state_feature_names
from so101_nexus._reproducibility import seed_everything
from so101_nexus._run_metadata import training_metadata
from so101_nexus.warp.base_env import _DELTA_ACTION_SCALE

try:
    from examples import _warp_training as _training
except ImportError:  # Direct script execution adds examples/ to sys.path.
    import _warp_training as _training  # ty: ignore[unresolved-import]


@dataclass
class Args:
    """BC-seeded PPO hyperparameters and run configuration (CleanRL layout)."""

    exp_name: str = "bc_ppo"
    seed: int = 1
    torch_deterministic: bool = True
    torch_deterministic_warn_only: bool = False
    cuda: bool = True

    env_id: str = "WarpPickLift-v1"
    """any registered ``Warp*-v1`` id (e.g. ``WarpPickAndPlace-v1``); see the module
    docstring's pick-and-place section for the validated flag set beyond ``WarpPickLift-v1``"""
    num_envs: int = 1024
    """number of GPU-parallel Warp worlds"""
    episode_length: int = 512
    """max steps per episode (truncation horizon)"""
    control_mode: str = "pd_joint_delta_pos"
    terminate_on_success: bool = False
    """default False = fixed-horizon (avoids the reach-farming trap; see ppo_warp.py's doc)"""
    success_bonus: float = 0.0
    """optional extra reward added on the step success is achieved (raw, pre-scale); ``0.0`` is
    correct for ``WarpPickLift-v1``, ``50.0`` is validated for ``WarpPickAndPlace-v1`` -- see
    the module docstring"""
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
    anneal_timesteps: int = 0
    """Decouples **both** `anneal_lr` and `ent_coef`'s annealing *horizon* from
    `total_timesteps` (`0` = anneal across the full `total_timesteps`, the historical
    behavior -- disabled by default, the `WarpPickLift-v1` recipe above is unaffected). Naively
    extending `total_timesteps` alone (schedule stretched across the longer horizon too)
    reliably backfires once a task's convergence is slow enough that exploration noise still
    matters mid-training. Setting this to a validated pace while raising `total_timesteps`
    well beyond it anneals both schedules on exactly that pace, then holds LR and `ent_coef` at
    their end-of-schedule values for the extended tail. Validated for `WarpPickAndPlace-v1` at
    `80_000_000` (paired with `total_timesteps=160_000_000`, `lr_min_frac=0.1` below) -- see
    the module docstring."""
    lr_min_frac: float = 0.0
    """Floors `anneal_lr`'s fraction at this value instead of letting it approach `0` at the
    anneal horizon (`0.0` = no floor, the historical behavior). Without a floor, the schedule
    anneals LR close enough to `0` that essentially no further optimization happens for any
    extended tail past `anneal_timesteps`. Validated default `0.1` (`eta_max/10`) for
    `WarpPickAndPlace-v1` when `anneal_timesteps > 0` -- see the module docstring."""
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None
    """KL early-stop threshold (None disables, matching CleanRL continuous PPO)"""

    norm_obs: bool = True
    norm_reward: bool = True
    hidden_dim: int = 256

    # demo seeding
    use_demos: bool = True
    """seed the actor from teleop demonstrations before online PPO"""
    demo_repo: str = "johnsutor/MuJoCoPickLift-v1"
    """HuggingFace dataset repo id with the seed rollouts (task-matched, e.g.
    ``johnsutor/MuJoCoPickAndPlace-v1`` for ``WarpPickAndPlace-v1``)"""
    demo_revision: str | None = None
    """dataset commit, tag, or branch; resolved to an immutable commit before loading"""
    bc_pretrain_updates: int = 2_000
    """supervised gradient steps regressing the actor mean onto demo actions, before PPO starts"""
    bc_pretrain_lr: float = 1e-3
    bc_batch_size: int = 256
    bc_coef: float = 0.1
    """persistent BC loss weight added to every PPO minibatch update (0 = pretrain-only, no
    ongoing demo influence); anchors the actor against drifting off the demos' successful
    grasp-lift manifold as online training explores"""
    bc_anneal_steps: int = 0
    """if >0, linearly decay bc_coef to 0 over this many env steps (0 = constant bc_coef)"""

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


class ObsNormalizer(_training.ObsNormalizer):
    """Preserve the BC recipe's output sanitization."""

    sanitize_outputs = True


class RewardScaler(_training.RewardScaler):
    """Preserve the BC recipe's output sanitization."""

    sanitize_outputs = True


layer_init = _training.layer_init
Agent = _training.Agent
_resolve_env_cls = _training._resolve_env_cls
_make_envs = _training._make_envs


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
):
    """Deterministic eval in the matching ``MuJoCo*`` backend (a transfer figure).

    Warp's render() is a no-op, so eval rollouts are rendered in the MuJoCo backend
    with the same config and control mode as training (``observations=None`` takes
    the all-observation default, otherwise the training layout is pinned); the saved
    Warp policy + obs-norm stats transfer directly (slight physics gap: Warp uses
    implicit/no-noslip). Returns (eval_metrics, frames) where frames is one episode's
    HxWx3 uint8 arrays (None when capture_video is False).
    """
    return _training.evaluate_mujoco(
        agent,
        obs_norm,
        device,
        env_id=env_id,
        control_mode=control_mode,
        episode_length=episode_length,
        eval_episodes=eval_episodes,
        seed=seed,
        capture_video=capture_video,
        observations=observations,
        config=config,
        sanitize_inputs=True,
    )


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
    checkpoint written before that was stored). It is a transfer figure: the saved
    policy and obs-norm stats transfer directly, with a slight physics gap versus Warp.
    Returns ``(metrics, video_path)`` where ``video_path`` is the written mp4, or
    ``None`` when ``capture_video`` is False.
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
        metadata_layout=False,
    )


def _save(
    agent, obs_norm, path, step, success, env_state_names=None, *, metadata=None, env_config=None
):
    """Save the BC recipe's inference snapshot and optional demonstration layout."""
    _training._save(
        agent,
        obs_norm,
        path,
        step,
        success,
        env_state_names,
        metadata=metadata,
        env_config=env_config,
    )


def resolve_demo_revision(demo_repo: str, revision: str | None = None) -> str:
    """Resolve a dataset revision to the immutable commit used for this run."""
    from huggingface_hub import HfApi

    if revision is not None and re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        return revision.lower()
    sha = HfApi().dataset_info(demo_repo, revision=revision).sha
    if sha is None or re.fullmatch(r"[0-9a-fA-F]{40}", sha) is None:
        raise ValueError(f"could not resolve {demo_repo}@{revision or 'main'} to a commit SHA")
    return sha.lower()


def demo_parquet_files(demo_repo: str, revision: str) -> list[str]:
    """Return the dataset's parquet files in a stable order."""
    from huggingface_hub import HfApi

    files = sorted(
        name
        for name in HfApi().list_repo_files(demo_repo, repo_type="dataset", revision=revision)
        if name.startswith("data/") and name.endswith(".parquet")
    )
    if not files:
        raise ValueError(f"dataset {demo_repo}@{revision} has no parquet data files")
    return files


def demo_observations(demo_repo: str, *, revision: str | None = None):
    """Return the observation layout a demo dataset declares for its state channel.

    A behaviour-cloned policy has to consume exactly the components the
    demonstrations recorded, so the dataset (not the env's current default) is
    what defines the layout for a BC run. Datasets recorded before a component
    joined the defaults simply produce the layout they were recorded with.
    """
    import json
    from pathlib import Path

    from huggingface_hub import hf_hub_download

    revision = resolve_demo_revision(demo_repo, revision)
    info = json.loads(
        Path(
            hf_hub_download(demo_repo, "meta/info.json", repo_type="dataset", revision=revision)
        ).read_text()
    )
    return observations_from_feature_names(
        info["features"]["observation.environment_state"]["names"]
    )


def load_demo_transitions(
    demo_repo: str,
    device: torch.device,
    *,
    control_dt: float,
    observations=None,
    revision: str | None = None,
    parquet_files: list[str] | None = None,
):
    """Download the HF teleop dataset and build flat (obs, action) BC transitions.

    ``obs`` matches the layout of ``observations`` (default: the layout the dataset
    itself declares, see :func:`demo_observations`). A layout that differs from the
    recording is re-laid by name via ``relabel_environment_state``, which reconstructs
    ``JointVelocities`` offline as a finite difference of the recorded
    ``JointPositions`` columns. ``control_dt`` is the simulated time between
    consecutive recorded frames (``env.unwrapped.control_dt``, one env step),
    NOT the dataset's wall-clock ``fps``: the teleop recorder sleeps to pace the
    operator but steps the simulation exactly once per frame, so dividing by
    ``1 / fps`` would scale every velocity by ``control_dt * fps``.

    ``action`` is the per-step joint-space delta between consecutive recorded
    joint states -- the realized motion, matching what ``pd_joint_delta_pos``
    needs to reproduce the trajectory -- normalized by ``_DELTA_ACTION_SCALE`` to
    the same ``[-1, 1]`` frame the env expects directly (no further env-side
    rescale, unlike an absolute-position control mode).

    Returns ``(obs, action)`` float32 tensors of shape ``(N, obs_dim)`` / ``(N, 6)``
    on ``device``.
    """
    import json
    from pathlib import Path

    import pandas as pd
    from huggingface_hub import hf_hub_download

    from so101_nexus import dataset_row_to_sim_qpos, relabel_environment_state

    env_state_key = "observation.environment_state"

    revision = resolve_demo_revision(demo_repo, revision)
    layout = list(
        observations
        if observations is not None
        else demo_observations(demo_repo, revision=revision)
    )
    if not layout:
        raise ValueError("target observation layout is empty; pass `observations`")

    print(f"[demos] downloading {demo_repo} ...", flush=True)
    parquet_files = parquet_files or demo_parquet_files(demo_repo, revision)
    parquet_paths = [
        hf_hub_download(demo_repo, name, repo_type="dataset", revision=revision)
        for name in parquet_files
    ]
    info = json.loads(
        Path(
            hf_hub_download(demo_repo, "meta/info.json", repo_type="dataset", revision=revision)
        ).read_text()
    )
    df = (
        pd.concat((pd.read_parquet(path) for path in parquet_paths), ignore_index=True)
        .sort_values("index", kind="stable")
        .reset_index(drop=True)
    )

    joints_raw = np.stack(df["observation.state"].to_numpy()).astype(np.float32)  # [N,6]
    ep_index = df["episode_index"].to_numpy()
    # `environment_state` is the env's full privileged state vector; its declared
    # per-column names let an older recording be mapped onto the current layout.
    obs_all = relabel_environment_state(
        np.stack(df[env_state_key].to_numpy()),
        info["features"][env_state_key]["names"],
        layout,
        dt=control_dt,
        episode_index=ep_index,
    )

    joints_rad = dataset_row_to_sim_qpos(joints_raw).astype(
        np.float32
    )  # SO101_GRIPPER_LIMITS_RAD default; used for the delta-action target below
    delta_scale = np.asarray(_DELTA_ACTION_SCALE, dtype=np.float32)

    obs_list, act_list = [], []
    for e in sorted(np.unique(ep_index)):
        idx = np.nonzero(ep_index == e)[0]
        s, t = idx[0], idx[-1]  # inclusive contiguous block, L = t - s + 1 rows
        ep_obs = obs_all[s:t]  # [L-1, obs_dim] -- drop the last row (no outgoing transition)
        ep_delta = joints_rad[s + 1 : t + 1] - joints_rad[s:t]  # [L-1, 6] realized motion
        obs_list.append(ep_obs)
        act_list.append(np.clip(ep_delta / delta_scale, -1.0, 1.0))
    obs = torch.as_tensor(np.concatenate(obs_list, axis=0), device=device)
    action = torch.as_tensor(np.concatenate(act_list, axis=0), device=device)
    print(f"[demos] built {obs.shape[0]} transitions from {len(obs_list)} episodes", flush=True)
    return obs, action


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
    anneal_timesteps=0,
    lr_min_frac=0.0,
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
    use_demos=True,
    demo_repo="johnsutor/MuJoCoPickLift-v1",
    demo_revision=None,
    bc_pretrain_updates=2_000,
    bc_pretrain_lr=1e-3,
    bc_batch_size=256,
    bc_coef=0.1,
    bc_anneal_steps=0,
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
    """Run BC-seeded PPO on the batched Warp env; return summary stats. Stays on ``device``.

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
    # Decoupled from num_updates when anneal_timesteps > 0: both anneal_lr and ent_coef reach
    # their end-of-schedule values at anneal_updates and hold there for any remaining updates,
    # instead of stretching across the full (possibly much longer) total_timesteps -- see
    # anneal_timesteps' docstring in Args. Default 0 reproduces the historical behavior
    # exactly (anneal_updates == num_updates).
    anneal_updates = max(
        1, (anneal_timesteps if anneal_timesteps > 0 else total_timesteps) // batch_size
    )

    seed_everything(
        seed,
        deterministic=torch_deterministic,
        deterministic_warn_only=torch_deterministic_warn_only,
    )
    dev = torch.device(device)
    np_rng = np.random.default_rng(seed)
    policy_rng = torch.Generator(device=dev).manual_seed(seed + 1)
    stagger_rng = torch.Generator(device=dev).manual_seed(seed + 2)
    demo_rng = torch.Generator(device=dev).manual_seed(seed + 3)

    # BC regresses the policy on recorded states, so the env must expose exactly the
    # layout the demos declare; without demos the env's own default is used.
    resolved_demo_revision = resolve_demo_revision(demo_repo, demo_revision) if use_demos else None
    demo_files = (
        demo_parquet_files(demo_repo, resolved_demo_revision)
        if resolved_demo_revision is not None
        else None
    )
    train_observations = (
        demo_observations(demo_repo, revision=resolved_demo_revision) if use_demos else None
    )
    if train_observations is not None:
        default_names = privileged_state_feature_names(
            _resolve_env_cls(env_id).default_config_cls().observations
        )
        demo_names = privileged_state_feature_names(train_observations)
        if demo_names != default_names:
            missing = [n for n in default_names if n not in demo_names]
            print(
                f"[demos] {demo_repo} declares a {len(demo_names)}-dim state layout; pinning the "
                f"env to it instead of the {len(default_names)}-dim default "
                f"(missing: {missing}). The checkpoint only loads against this layout; "
                "re-record the demos to train on the current default.",
                flush=True,
            )
    envs = _make_envs(
        env_id,
        num_envs,
        dev,
        seed,
        control_mode=control_mode,
        episode_length=episode_length,
        terminate_on_success=terminate_on_success,
        observations=train_observations,
    )
    names = privileged_state_feature_names(envs.unwrapped.config.observations)
    run_metadata = training_metadata(
        device=dev,
        env_config=envs.unwrapped.config,
        env_id=env_id,
        control_mode=control_mode,
        episode_length=episode_length,
        hidden_dim=hidden_dim,
        seed=seed,
        demo_repo=demo_repo if use_demos else None,
        demo_revision=resolved_demo_revision,
        demo_files=demo_files,
        run_config=run_config,
    )
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    act_dim = int(np.prod(envs.single_action_space.shape))

    agent = Agent(obs_dim, act_dim, hidden_dim).to(dev)
    optimizer = torch.optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)
    obs_norm = ObsNormalizer(obs_dim, dev, enabled=norm_obs)
    rew_scaler = RewardScaler(num_envs, dev, gamma, enabled=norm_reward)

    demo_obs = demo_act = None
    if use_demos:
        demo_obs, demo_act = load_demo_transitions(
            demo_repo,
            dev,
            observations=envs.unwrapped.config.observations,
            control_dt=envs.unwrapped.control_dt,
            revision=resolved_demo_revision,
            parquet_files=demo_files,
        )
        obs_norm(demo_obs, update=True)  # fit running stats on demo obs before BC regression
        if bc_pretrain_updates > 0:
            print(f"[demos] BC-pretraining actor for {bc_pretrain_updates} updates ...", flush=True)
            t0 = time.time()
            bc_optim = torch.optim.Adam(agent.actor_mean.parameters(), lr=bc_pretrain_lr)
            n_demo = demo_obs.shape[0]
            for u in range(bc_pretrain_updates):
                idx = torch.randint(0, n_demo, (bc_batch_size,), device=dev, generator=demo_rng)
                pred = agent.actor_mean(obs_norm(demo_obs[idx], update=False))
                pretrain_loss = F.mse_loss(pred, demo_act[idx])
                bc_optim.zero_grad(set_to_none=True)
                if torch.isfinite(pretrain_loss):
                    pretrain_loss.backward()
                    nn.utils.clip_grad_norm_(agent.actor_mean.parameters(), max_grad_norm)
                    bc_optim.step()
                if writer is not None and (u + 1) % 100 == 0:
                    writer.add_scalar("pretrain/bc_loss", pretrain_loss.item(), u + 1)
            print(f"[demos] BC pretrain done in {time.time() - t0:.1f}s", flush=True)

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

    pg_loss = v_loss = entropy_loss = bc_loss = torch.tensor(0.0)
    nonfinite_updates = 0
    approx_kl = torch.tensor(0.0)
    clipfracs: list[torch.Tensor] = []
    succ_rate = mean_ret = mean_len = 0.0

    for update in range(1, num_updates + 1):
        anneal_eff_update = min(update, anneal_updates)
        if anneal_lr:
            frac = max(lr_min_frac, 1.0 - (anneal_eff_update - 1.0) / anneal_updates)
            optimizer.param_groups[0]["lr"] = frac * learning_rate
        ent_now = ent_coef + (ent_coef_final - ent_coef) * (
            (anneal_eff_update - 1.0) / anneal_updates
        )
        if bc_anneal_steps > 0:
            bc_coef_now = bc_coef * max(0.0, 1.0 - global_step / bc_anneal_steps)
        else:
            bc_coef_now = bc_coef

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

                # Persistent BC anchor: pulls the actor mean toward demo actions on a fresh
                # demo minibatch every PPO minibatch step, independent of the PPO batch above.
                bc_loss = torch.tensor(0.0, device=dev)
                if (
                    use_demos
                    and bc_coef_now > 0.0
                    and demo_obs is not None
                    and demo_act is not None
                ):
                    d_idx = torch.randint(
                        0, demo_obs.shape[0], (bc_batch_size,), device=dev, generator=demo_rng
                    )
                    d_pred = agent.actor_mean(obs_norm(demo_obs[d_idx], update=False))
                    bc_loss = F.mse_loss(d_pred, demo_act[d_idx])
                    loss = loss + bc_coef_now * bc_loss

                optimizer.zero_grad(set_to_none=True)
                # A NaN/Inf loss (from a diverging Warp world slipping past the
                # sanitization above) must not reach the optimizer: backward()
                # would NaN-poison every parameter's gradient, and step() would
                # then permanently corrupt the policy weights.
                if torch.isfinite(loss):
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                    optimizer.step()
                else:
                    # Defense in depth on top of the obs/reward sanitization above: never let
                    # a non-finite loss (e.g. from a pathological minibatch) reach the
                    # optimizer, which would otherwise permanently NaN Adam's moment estimates.
                    nonfinite_updates += 1
                    print(
                        f"[warn] non-finite loss at step {global_step}, skipping update "
                        f"({nonfinite_updates} so far)",
                        flush=True,
                    )

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
                    names,
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
                "charts/bc_coef": bc_coef_now,
                "charts/best_success": max(best_success, 0.0),
                "losses/policy_loss": pg_loss.item(),
                "losses/value_loss": v_loss.item(),
                "losses/entropy": entropy_loss.item(),
                "losses/bc_loss": bc_loss.item(),
                "losses/approx_kl": approx_kl.item(),
                "losses/clipfrac": float(torch.stack(clipfracs).mean()) if clipfracs else 0.0,
                "charts/nonfinite_updates": nonfinite_updates,
            }
            if writer is not None:
                for k, v in metrics.items():
                    writer.add_scalar(k, v, global_step)
            if log:
                print(
                    f"[step {global_step}] success={succ_rate:.3f} return={mean_ret:.1f} "
                    f"ent={ent_now:.4f} bc={bc_coef_now:.4f} SPS={sps}",
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
                    observations=train_observations,
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
            names,
            metadata=run_metadata,
            env_config=envs.unwrapped.config,
        )

    envs.close()
    return {
        "iterations": num_updates,
        "policy_loss": float(pg_loss.item()),
        "value_loss": float(v_loss.item()),
        "bc_loss": float(bc_loss.item()),
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
        anneal_timesteps=args.anneal_timesteps,
        lr_min_frac=args.lr_min_frac,
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
        use_demos=args.use_demos,
        demo_repo=args.demo_repo,
        demo_revision=args.demo_revision,
        bc_pretrain_updates=args.bc_pretrain_updates,
        bc_pretrain_lr=args.bc_pretrain_lr,
        bc_batch_size=args.bc_batch_size,
        bc_coef=args.bc_coef,
        bc_anneal_steps=args.bc_anneal_steps,
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
