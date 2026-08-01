"""Tests for ``examples/eval_warp.py``, the deterministic checkpoint evaluator.

``eval_warp.py`` is documented (and used by both Colab notebooks) to evaluate
checkpoints from ``ppo_warp.py`` *and* ``bc_ppo_warp.py``. The latter pins the
observation layout its demo dataset declares, which is narrower than the env
default whenever a component joined the defaults after the demos were recorded,
so the evaluator has to rebuild the env from the layout the checkpoint records
rather than the env's current default.
"""

import importlib

import pytest
import torch

from so101_nexus import privileged_state_feature_names

pytestmark = pytest.mark.warp


def _default_observations(env_id="WarpPickLift-v1"):
    mod = importlib.import_module("examples.ppo_warp")
    return list(mod._resolve_env_cls(env_id).default_config_cls().observations)


def _obs_dim(env_id, observations):
    mod = importlib.import_module("examples.ppo_warp")
    envs = mod._make_envs(env_id, 1, torch.device("cpu"), 0, observations=observations)
    try:
        return int(envs.single_observation_space.shape[0])
    finally:
        envs.close()


def test_make_envs_pins_the_requested_observation_layout():
    """A pinned layout must narrow the env's observation width, not be ignored."""
    env_id = "WarpPickLift-v1"
    default = _default_observations(env_id)
    narrowed = default[:-1]

    default_dim = _obs_dim(env_id, None)
    narrowed_dim = _obs_dim(env_id, narrowed)

    assert default_dim == len(privileged_state_feature_names(default))
    assert narrowed_dim == len(privileged_state_feature_names(narrowed))
    assert narrowed_dim < default_dim


def _write_checkpoint(path, obs_dim, act_dim, hidden_dim, env_state_names):
    mod = importlib.import_module("examples.ppo_warp")
    agent = mod.Agent(obs_dim, act_dim, hidden_dim)
    torch.save(
        {
            "model": agent.state_dict(),
            "obs_mean": torch.zeros(obs_dim),
            "obs_var": torch.ones(obs_dim),
            "step": 0,
            "success": 0.0,
            "env_state_names": env_state_names,
        },
        path,
    )


@pytest.mark.parametrize("pinned", [True, False])
def test_eval_warp_rebuilds_the_env_from_the_checkpoint_layout(tmp_path, monkeypatch, pinned):
    """A checkpoint trained on a narrower (demo-pinned) layout must still evaluate.

    Regression: ``eval_warp.py`` built the env from the env's default layout and
    crashed with a ``size mismatch`` ``RuntimeError`` on every ``bc_ppo_warp.py``
    checkpoint whose demos predate a default observation component.
    """
    env_id = "WarpPickLift-v1"
    hidden_dim = 8
    observations = _default_observations(env_id)
    if pinned:
        observations = observations[:-1]
    names = privileged_state_feature_names(observations)

    checkpoint = tmp_path / "agent.pt"
    # Width comes from the names, not from `_make_envs`, so the pinned case fails
    # on the real `load_state_dict` size mismatch when the evaluator ignores them.
    _write_checkpoint(checkpoint, len(names), 6, hidden_dim, names if pinned else None)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        "sys.argv",
        [
            "eval_warp.py",
            "--checkpoint",
            str(checkpoint),
            "--env-id",
            env_id,
            "--num-envs",
            "2",
            "--episode-length",
            "2",
            "--hidden-dim",
            str(hidden_dim),
        ],
    )

    importlib.import_module("examples.eval_warp").main()
