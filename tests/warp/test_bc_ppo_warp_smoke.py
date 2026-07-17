import numpy as np
import pytest

pytestmark = pytest.mark.warp


def test_bc_ppo_warp_defaults_use_demos_with_persistent_bc_loss():
    import importlib

    mod = importlib.import_module("examples.bc_ppo_warp")
    args = mod.Args()

    assert args.use_demos is True
    assert args.demo_repo == "johnsutor/MuJoCoPickLift-v1"
    assert args.bc_pretrain_updates > 0
    assert args.bc_coef > 0.0
    assert args.control_mode == "pd_joint_delta_pos"  # unchanged from ppo_warp.py


def test_bc_ppo_warp_shares_ppo_warp_decisive_defaults():
    """The PPO scaffolding (fixed-horizon, entropy schedule, optimizer budget) must
    stay identical to ppo_warp.py's proven recipe -- BC-seeding is additive."""
    import importlib

    mod = importlib.import_module("examples.bc_ppo_warp")
    args = mod.Args()

    assert args.terminate_on_success is False
    assert args.ent_coef == 0.03
    assert args.ent_coef_final == 0.005
    assert args.num_minibatches == 32
    assert args.update_epochs == 10
    assert args.max_grad_norm == 0.5
    assert args.target_kl is None


def test_bc_ppo_warp_load_demo_transitions_shapes_and_units():
    import importlib

    import torch

    mod = importlib.import_module("examples.bc_ppo_warp")
    obs, action = mod.load_demo_transitions("johnsutor/MuJoCoPickLift-v1", torch.device("cpu"))

    assert obs.ndim == 2
    assert obs.shape[1] == 24  # joints(6) + ee_pose(7) + grasp(1) + obj_pose(7) + obj_offset(3)
    assert action.shape == (obs.shape[0], 6)
    assert obs.shape[0] > 0
    # deltas are normalized into the env's native [-1, 1] pd_joint_delta_pos frame
    assert float(action.min()) >= -1.0 - 1e-6
    assert float(action.max()) <= 1.0 + 1e-6
    assert torch.isfinite(obs).all()
    assert torch.isfinite(action).all()


def test_bc_ppo_warp_delta_action_matches_consecutive_joint_difference():
    """Regression test on the unit conversion: the demo action for a transition must
    equal the realized joint delta (not the recorded absolute-position `action`
    column), normalized by `_DELTA_ACTION_SCALE` and clipped to [-1, 1]."""
    import importlib

    import pandas as pd
    import torch

    from so101_nexus import dataset_row_to_sim_qpos
    from so101_nexus.warp.base_env import _DELTA_ACTION_SCALE

    mod = importlib.import_module("examples.bc_ppo_warp")

    from huggingface_hub import hf_hub_download

    pq = hf_hub_download(
        "johnsutor/MuJoCoPickLift-v1", "data/chunk-000/file-000.parquet", repo_type="dataset"
    )
    df = pd.read_parquet(pq).sort_values("index").reset_index(drop=True)
    df = df[df["episode_index"] == df["episode_index"].iloc[0]].reset_index(drop=True)
    joints_rad = dataset_row_to_sim_qpos(
        np.stack(df["observation.state"].to_numpy()).astype(np.float32)
    )
    expected_first = np.clip(
        (joints_rad[1] - joints_rad[0]) / np.asarray(_DELTA_ACTION_SCALE, dtype=np.float32),
        -1.0,
        1.0,
    )

    _, action = mod.load_demo_transitions("johnsutor/MuJoCoPickLift-v1", torch.device("cpu"))
    np.testing.assert_allclose(action[0].numpy(), expected_first, atol=1e-5)


def test_bc_ppo_warp_pretrain_clips_actor_gradients():
    """Regression test: BC-pretrain's own optimizer step must clip gradients like
    the main PPO loop does. The pretrain loop has a separate backward/step cycle
    (its own ``bc_optim``), so PPO's ``clip_grad_norm_`` call does not cover it."""
    import importlib

    from torch import nn

    mod = importlib.import_module("examples.bc_ppo_warp")
    calls = []
    real_clip = nn.utils.clip_grad_norm_

    def spy_clip(parameters, max_norm, *args, **kwargs):
        calls.append(max_norm)
        return real_clip(parameters, max_norm, *args, **kwargs)

    from unittest import mock

    with mock.patch.object(nn.utils, "clip_grad_norm_", side_effect=spy_clip):
        mod.train(
            num_envs=8,
            num_steps=8,
            total_timesteps=8 * 8 * 2,
            num_minibatches=4,
            device="cpu",
            seed=0,
            use_demos=True,
            bc_pretrain_updates=5,
            bc_batch_size=16,
            bc_coef=0.0,  # isolate the pretrain-phase clip call from the online one
            max_grad_norm=0.5,
        )
    assert len(calls) >= 5  # one clip call per pretrain update, plus PPO's own
    assert all(c == 0.5 for c in calls)


def test_bc_ppo_warp_short_run_finite_with_demos():
    import importlib

    import torch

    mod = importlib.import_module("examples.bc_ppo_warp")
    stats = mod.train(
        num_envs=8,
        num_steps=8,
        total_timesteps=8 * 8 * 2,  # two iterations
        num_minibatches=4,
        device="cpu",
        seed=0,
        use_demos=True,
        bc_pretrain_updates=10,
        bc_batch_size=16,
        bc_coef=0.1,
    )
    assert torch.isfinite(torch.tensor(stats["policy_loss"]))
    assert torch.isfinite(torch.tensor(stats["value_loss"]))
    assert torch.isfinite(torch.tensor(stats["bc_loss"]))
    assert stats["bc_loss"] > 0.0  # persistent BC term actually fired
    assert stats["iterations"] == 2


def test_bc_ppo_warp_short_run_finite_without_demos_matches_ppo_warp_shape():
    """``use_demos=False`` must behave like a plain PPO run (bc_loss stays zero, no
    network access), keeping this file a strict superset of ppo_warp.py's recipe."""
    import importlib

    import torch

    mod = importlib.import_module("examples.bc_ppo_warp")
    stats = mod.train(
        num_envs=8,
        num_steps=8,
        total_timesteps=8 * 8 * 2,
        num_minibatches=4,
        device="cpu",
        seed=0,
        use_demos=False,
    )
    assert torch.isfinite(torch.tensor(stats["policy_loss"]))
    assert stats["bc_loss"] == 0.0
    assert stats["iterations"] == 2


def test_bc_ppo_warp_same_seed_cpu_short_runs_are_reproducible():
    """Same-seed CPU runs (including demo download/BC-pretrain) must return
    identical deterministic scalar stats."""
    import importlib
    import math

    mod = importlib.import_module("examples.bc_ppo_warp")
    kwargs = {
        "num_envs": 8,
        "num_steps": 8,
        "total_timesteps": 8 * 8 * 2,
        "num_minibatches": 4,
        "device": "cpu",
        "seed": 123,
        "use_demos": True,
        "bc_pretrain_updates": 10,
        "bc_batch_size": 16,
        "bc_coef": 0.1,
        "capture_video": False,
        "eval_freq": 0,
        "log": False,
    }

    first = mod.train(**kwargs)
    second = mod.train(**kwargs)

    keys = (
        "iterations",
        "episodes",
        "policy_loss",
        "value_loss",
        "bc_loss",
        "success_rate",
        "best_success",
    )
    for key in keys:
        assert second[key] == first[key], key
    if math.isnan(first["mean_return"]):
        assert math.isnan(second["mean_return"])
    else:
        assert second["mean_return"] == first["mean_return"]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {"num_envs": 2, "num_steps": 2, "total_timesteps": 4, "num_minibatches": 5},
            "num_minibatches",
        ),
        ({"num_envs": 3, "num_steps": 2, "total_timesteps": 6, "num_minibatches": 4}, "divisible"),
        (
            {"num_envs": 2, "num_steps": 2, "total_timesteps": 3, "num_minibatches": 2},
            "total_timesteps",
        ),
    ],
)
def test_bc_ppo_warp_rejects_invalid_batch_args_before_env_construction(monkeypatch, kwargs, match):
    import importlib

    mod = importlib.import_module("examples.bc_ppo_warp")

    def fail_make_envs(*_args, **_kwargs):
        raise AssertionError("_make_envs should not be called for invalid batch args")

    monkeypatch.setattr(mod, "_make_envs", fail_make_envs)
    with pytest.raises(ValueError, match=match):
        mod.train(device="cpu", use_demos=False, **kwargs)


def test_running_mean_std_update_sanitizes_nan_and_inf():
    """A NaN/Inf row in a batch must not permanently poison the running
    mean/variance accumulator (a corrupted stat mixes NaN into every future
    ``update()`` via the Welford recurrence): the vulnerability that could
    crash a long ``bc_ppo_warp.py`` run with ``ValueError: Normal(loc=NaN,
    ...)``. Same ``RunningMeanStd`` code as ``ppo_warp.py`` (copied verbatim),
    tested again here since the two scripts are standalone copies that could
    drift independently.
    """
    import torch

    import examples.bc_ppo_warp as mod

    rms = mod.RunningMeanStd((3,), "cpu")
    poisoned = torch.tensor(
        [[float("nan"), 1.0, float("inf")], [1.0, float("-inf"), 2.0]], dtype=torch.float32
    )
    rms.update(poisoned)
    assert torch.isfinite(rms.mean).all()
    assert torch.isfinite(rms.var).all()

    rms.update(torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=torch.float32))
    assert torch.isfinite(rms.mean).all()
    assert torch.isfinite(rms.var).all()


def test_obs_normalizer_sanitizes_nan_observation():
    import torch

    import examples.bc_ppo_warp as mod

    norm = mod.ObsNormalizer(3, "cpu")
    obs = torch.tensor([[float("nan"), float("inf"), 1.0]])
    out = norm(obs, update=True)
    assert torch.isfinite(out).all()
    assert bool((out >= -10.0).all() and (out <= 10.0).all())


def test_reward_scaler_sanitizes_nan_reward():
    import torch

    import examples.bc_ppo_warp as mod

    scaler = mod.RewardScaler(2, "cpu", gamma=0.99)
    reward = torch.tensor([float("nan"), float("inf")])
    done = torch.zeros(2)
    out = scaler(reward, done)
    assert torch.isfinite(out).all()
    assert torch.isfinite(scaler.rms.mean).all()
    assert torch.isfinite(scaler.rms.var).all()


def test_bc_ppo_warp_survives_nan_reward_from_one_env_step(monkeypatch):
    """End-to-end regression: a single NaN reward from one parallel Warp world
    (injected at the exact ``envs.step()`` boundary ``train()`` reads) must not
    propagate into the shared reward-scaler stats, the loss, or the optimizer
    step -- training must finish with finite losses instead of eventually
    crashing on a poisoned ``RunningMeanStd``. ``use_demos=False`` keeps this
    test network-free. See docs/superpowers/plans/
    2026-07-16-pick-grasp-potential-shaping.md.
    """
    import importlib

    import torch

    mod = importlib.import_module("examples.bc_ppo_warp")
    real_make_envs = mod._make_envs
    poisoned = {"done": False}

    def make_envs_with_one_nan_step(*args, **kwargs):
        envs = real_make_envs(*args, **kwargs)
        real_step = envs.step

        def step(action):
            obs, reward, terminated, truncated, info = real_step(action)
            if not poisoned["done"]:
                poisoned["done"] = True
                reward = reward.clone()
                reward[0] = float("nan")
            return obs, reward, terminated, truncated, info

        envs.step = step
        return envs

    monkeypatch.setattr(mod, "_make_envs", make_envs_with_one_nan_step)
    stats = mod.train(
        num_envs=8,
        num_steps=8,
        total_timesteps=8 * 8 * 2,  # two iterations
        num_minibatches=4,
        device="cpu",
        seed=0,
        use_demos=False,
    )
    assert torch.isfinite(torch.tensor(stats["policy_loss"]))
    assert torch.isfinite(torch.tensor(stats["value_loss"]))
    assert stats["iterations"] == 2
