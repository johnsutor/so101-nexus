import pytest

pytestmark = pytest.mark.warp


def test_ppo_warp_default_budget_preserves_picklift_annealing_horizon():
    import importlib

    mod = importlib.import_module("examples.ppo_warp")

    assert mod.Args().total_timesteps == 200_000_000


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
