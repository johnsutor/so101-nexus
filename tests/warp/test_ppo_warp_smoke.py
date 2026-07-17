import pytest

pytestmark = pytest.mark.warp


def test_ppo_warp_default_budget_matches_validated_picklift_recipe():
    import importlib

    mod = importlib.import_module("examples.ppo_warp")

    assert mod.Args().total_timesteps == 30_000_000


def test_ppo_warp_default_entropy_bonus_is_strong_warm_start_with_floor():
    import importlib

    mod = importlib.import_module("examples.ppo_warp")

    args = mod.Args()

    assert args.ent_coef == 0.03
    assert args.ent_coef_final == 0.005


def test_ppo_warp_defaults_use_cleanrl_optimizer_budget():
    import importlib

    mod = importlib.import_module("examples.ppo_warp")

    args = mod.Args()

    assert args.num_minibatches == 32
    assert args.update_epochs == 10
    assert args.max_grad_norm == 0.5
    assert args.target_kl is None


def test_ppo_default_entropy_bonus_is_disabled():
    import importlib

    mod = importlib.import_module("examples.ppo")

    assert mod.Args().ent_coef == 0.0


def test_ppo_warp_short_run_finite():
    import importlib

    import torch

    mod = importlib.import_module("examples.ppo_warp")
    stats = mod.train(
        num_envs=8,
        num_steps=8,
        total_timesteps=8 * 8 * 2,  # two iterations
        num_minibatches=4,
        device="cpu",
        seed=0,
    )
    assert torch.isfinite(torch.tensor(stats["policy_loss"]))
    assert torch.isfinite(torch.tensor(stats["value_loss"]))
    assert stats["iterations"] == 2


def test_ppo_warp_same_seed_cpu_short_runs_are_reproducible():
    """Same-seed CPU runs must return identical deterministic scalar stats."""
    import importlib
    import math

    mod = importlib.import_module("examples.ppo_warp")
    kwargs = {
        "num_envs": 8,
        "num_steps": 8,
        "total_timesteps": 8 * 8 * 2,  # two iterations
        "num_minibatches": 4,
        "device": "cpu",
        "seed": 123,
        "capture_video": False,
        "eval_freq": 0,
        "log": False,
    }

    first = mod.train(**kwargs)
    second = mod.train(**kwargs)

    for key in (
        "iterations",
        "episodes",
        "policy_loss",
        "value_loss",
        "success_rate",
        "best_success",
    ):
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
def test_ppo_warp_rejects_invalid_batch_args_before_env_construction(monkeypatch, kwargs, match):
    import importlib

    mod = importlib.import_module("examples.ppo_warp")

    def fail_make_envs(*_args, **_kwargs):
        raise AssertionError("_make_envs should not be called for invalid batch args")

    monkeypatch.setattr(mod, "_make_envs", fail_make_envs)
    with pytest.raises(ValueError, match=match):
        mod.train(device="cpu", **kwargs)


@pytest.mark.parametrize(
    "env_id",
    [
        "WarpTouch-v1",
        "WarpLookAt-v1",
        "WarpMove-v1",
        "WarpPickLift-v1",
        "WarpPickAndPlace-v1",
    ],
)
def test_ppo_warp_runs_on_every_warp_env(env_id):
    """The CleanRL PPO trainer completes two update iterations on every Warp env id
    and reports finite policy/value losses -- proving the recipe is env-agnostic, not
    just tuned for the default WarpPickLift-v1."""
    import importlib

    import torch

    mod = importlib.import_module("examples.ppo_warp")
    stats = mod.train(
        env_id=env_id,
        num_envs=8,
        num_steps=8,
        total_timesteps=8 * 8 * 2,  # two iterations
        num_minibatches=4,
        device="cpu",
        seed=0,
    )
    assert torch.isfinite(torch.tensor(stats["policy_loss"]))
    assert torch.isfinite(torch.tensor(stats["value_loss"]))
    assert stats["iterations"] == 2


def test_running_mean_std_update_sanitizes_nan_and_inf():
    """A NaN/Inf row in a batch must not permanently poison the running
    mean/variance accumulator (a corrupted stat mixes NaN into every future
    ``update()`` via the Welford recurrence). Regression coverage for the
    NaN-poisoning vulnerability flagged for ``examples/bc_ppo_warp.py``
    (equally present in ``ppo_warp.py``, which the two scripts share
    verbatim): one diverging Warp world in a batched rollout could otherwise
    crash a long run with ``ValueError: Normal(loc=NaN, ...)``.
    """
    import torch

    import examples.ppo_warp as mod

    rms = mod.RunningMeanStd((3,), "cpu")
    poisoned = torch.tensor(
        [[float("nan"), 1.0, float("inf")], [1.0, float("-inf"), 2.0]], dtype=torch.float32
    )
    rms.update(poisoned)
    assert torch.isfinite(rms.mean).all()
    assert torch.isfinite(rms.var).all()

    # A poisoned accumulator would keep producing NaN/Inf forever; a clean
    # update afterwards must still land on finite, sane stats.
    rms.update(torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=torch.float32))
    assert torch.isfinite(rms.mean).all()
    assert torch.isfinite(rms.var).all()


def test_obs_normalizer_sanitizes_nan_observation():
    import torch

    import examples.ppo_warp as mod

    norm = mod.ObsNormalizer(3, "cpu")
    obs = torch.tensor([[float("nan"), float("inf"), 1.0]])
    out = norm(obs, update=True)
    assert torch.isfinite(out).all()
    assert bool((out >= -10.0).all() and (out <= 10.0).all())


def test_reward_scaler_sanitizes_nan_reward():
    import torch

    import examples.ppo_warp as mod

    scaler = mod.RewardScaler(2, "cpu", gamma=0.99)
    reward = torch.tensor([float("nan"), float("inf")])
    done = torch.zeros(2)
    out = scaler(reward, done)
    assert torch.isfinite(out).all()
    assert torch.isfinite(scaler.rms.mean).all()
    assert torch.isfinite(scaler.rms.var).all()


def test_ppo_warp_survives_nan_reward_from_one_env_step(monkeypatch):
    """End-to-end regression: a single NaN reward from one parallel Warp world
    (injected at the exact ``envs.step()`` boundary ``train()`` reads) must not
    propagate into the shared reward-scaler stats, the loss, or the optimizer
    step -- training must finish with finite losses instead of eventually
    crashing on a poisoned ``RunningMeanStd``. See docs/superpowers/plans/
    2026-07-16-pick-grasp-potential-shaping.md.
    """
    import importlib

    import torch

    mod = importlib.import_module("examples.ppo_warp")
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
    )
    assert torch.isfinite(torch.tensor(stats["policy_loss"]))
    assert torch.isfinite(torch.tensor(stats["value_loss"]))
    assert stats["iterations"] == 2
