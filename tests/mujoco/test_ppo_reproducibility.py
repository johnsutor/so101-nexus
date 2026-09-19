"""Reproducibility checks for the scalar MuJoCo PPO example."""

from types import SimpleNamespace

import numpy as np
import torch


def test_video_evaluation_does_not_advance_training_rng(monkeypatch):
    from examples import ppo

    class Env:
        spec = SimpleNamespace(max_episode_steps=1)

        def reset(self, seed=None):
            return np.zeros(3, dtype=np.float32), {}

        def render(self):
            return np.zeros((2, 2, 3), dtype=np.uint8)

        def step(self, action):
            return np.zeros(3, dtype=np.float32), 0.0, True, False, {}

        def close(self):
            pass

    class Wandb:
        class Video:
            def __init__(self, *args, **kwargs):
                pass

        @staticmethod
        def log(*args, **kwargs):
            pass

    envs = SimpleNamespace(
        single_observation_space=SimpleNamespace(shape=(3,)),
        single_action_space=SimpleNamespace(shape=(2,)),
    )
    agent = ppo.Agent(envs)
    args = SimpleNamespace(
        track=True,
        wandb_log_video=True,
        video_log_interval=1,
        env_id="test",
        video_fps=10,
    )
    monkeypatch.setattr(ppo, "make_video_eval_env", lambda env_id: Env())

    torch.manual_seed(123)
    before = torch.random.get_rng_state().clone()
    ppo.maybe_log_wandb_rollout_video(
        args=args,
        agent=agent,
        device=torch.device("cpu"),
        wandb_module=Wandb(),
        global_step=1,
        run_seed=123,
    )

    assert torch.equal(torch.random.get_rng_state(), before)
