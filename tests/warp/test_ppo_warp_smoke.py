import pytest

pytestmark = pytest.mark.warp


@pytest.mark.parametrize("module", ["examples.ppo_warp", "examples.bc_ppo_warp"])
def test_episode_statistics_span_rollouts(module, monkeypatch):
    import importlib

    import torch

    mod = importlib.import_module(module)
    make_envs = mod._make_envs
    metrics = {}

    def make(*args, **kwargs):
        env = make_envs(*args, **kwargs)
        original_step = env.step

        def step(actions):
            obs, _, terminated, truncated, info = original_step(actions)
            info["success"] = torch.tensor([True, False])
            return obs, torch.tensor([1.0, 2.0]), terminated, truncated, info

        env.step = step
        return env

    class Writer:
        def add_scalar(self, name, value, step):
            metrics[name] = value

    monkeypatch.setattr(mod, "_make_envs", make)
    stats = mod.train(
        num_envs=2,
        num_steps=5,
        total_timesteps=20,
        num_minibatches=1,
        update_epochs=1,
        episode_length=3,
        stagger_resets=False,
        device="cpu",
        writer=Writer(),
        log=False,
    )
    assert stats["episodes"] == 6
    assert stats["mean_return"] == 4.5
    assert stats["success_rate"] == 0.5
    assert metrics["charts/episodic_length"] == 3
    assert metrics["charts/hold_frac"] == 0.5


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


def test_ppo_warp_same_seed_cpu_short_runs_are_reproducible(tmp_path):
    """Same-seed CPU runs must return identical stats and inference tensors."""
    import importlib
    import math

    import torch

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

    first = mod.train(**kwargs, save_dir=str(tmp_path / "first"))
    second = mod.train(**kwargs, save_dir=str(tmp_path / "second"))

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
    first_checkpoint = torch.load(tmp_path / "first/agent.pt", weights_only=False)
    second_checkpoint = torch.load(tmp_path / "second/agent.pt", weights_only=False)
    assert first_checkpoint["model"].keys() == second_checkpoint["model"].keys()
    for name, tensor in first_checkpoint["model"].items():
        torch.testing.assert_close(tensor, second_checkpoint["model"][name], rtol=0, atol=0)
    torch.testing.assert_close(
        first_checkpoint["obs_mean"], second_checkpoint["obs_mean"], rtol=0, atol=0
    )
    torch.testing.assert_close(
        first_checkpoint["obs_var"], second_checkpoint["obs_var"], rtol=0, atol=0
    )


def test_ppo_warp_same_seed_is_reproducible_across_processes(tmp_path):
    import subprocess
    import sys

    import torch

    script = """
import sys
from examples.ppo_warp import train
train(num_envs=4, num_steps=4, total_timesteps=16, num_minibatches=2,
      update_epochs=1, device='cpu', seed=321, capture_video=False,
      eval_freq=0, log=False, save_dir=sys.argv[1])
"""
    paths = [tmp_path / "process_a", tmp_path / "process_b"]
    for path in paths:
        subprocess.run([sys.executable, "-c", script, str(path)], check=True)

    first = torch.load(paths[0] / "agent.pt", weights_only=False)
    second = torch.load(paths[1] / "agent.pt", weights_only=False)
    for name, tensor in first["model"].items():
        torch.testing.assert_close(tensor, second["model"][name], rtol=0, atol=0)
    torch.testing.assert_close(first["obs_mean"], second["obs_mean"], rtol=0, atol=0)
    torch.testing.assert_close(first["obs_var"], second["obs_var"], rtol=0, atol=0)


def test_inference_checkpoint_records_normalization_and_metadata(tmp_path):
    import importlib

    import torch

    mod = importlib.import_module("examples.ppo_warp")
    agent = mod.Agent(3, 2, 8)
    obs_norm = mod.ObsNormalizer(3, torch.device("cpu"), enabled=False)
    path = tmp_path / "agent.pt"

    config = mod._resolve_env_cls("WarpPickLift-v1").default_config_cls(spawn_max_radius=0.22)
    mod._save(
        agent,
        obs_norm,
        str(path),
        10,
        0.5,
        metadata={"seed": 7},
        env_config=config,
    )
    checkpoint = torch.load(path, weights_only=False)

    assert checkpoint["checkpoint_type"] == "inference"
    assert checkpoint["checkpoint_version"] == 1
    assert checkpoint["norm_obs"] is False
    assert checkpoint["metadata"] == {"seed": 7}
    assert checkpoint["env_config"].spawn_max_radius == 0.22


def test_training_metadata_records_resolved_config_and_runtime_versions():
    import torch

    from so101_nexus import JointPositions, PickConfig
    from so101_nexus._run_metadata import training_metadata

    config = PickConfig(
        spawn_max_radius=0.22,
        observations=[JointPositions()],
        reset_settle_frames=0,
    )
    metadata = training_metadata(
        device=torch.device("cpu"),
        env_config=config,
        seed=7,
        run_config={"learning_rate": 3e-4, "num_envs": 8, "norm_obs": False},
    )

    assert metadata["seed"] == 7
    assert metadata["env_config"]["spawn_max_radius"] == 0.22
    assert metadata["env_config"]["reset_settle_frames"] == 0
    assert metadata["env_config"]["observations"] == [{"__type__": "JointPositions"}]
    assert set(vars(config)) <= set(metadata["env_config"])
    assert metadata["run_config"] == {
        "learning_rate": 3e-4,
        "num_envs": 8,
        "norm_obs": False,
    }
    assert set(metadata["versions"]) == {
        "python",
        "numpy",
        "torch",
        "mujoco",
        "mujoco-warp",
        "warp-lang",
        "so101-nexus",
    }


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
    crashing on a poisoned ``RunningMeanStd``.
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
