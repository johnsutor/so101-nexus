"""Tests for the Colab sample-rollout helper in ``examples/ppo_warp.py``.

The helper renders one deterministic MuJoCo rollout of a saved Warp PPO policy and
writes an mp4. Two tests cover it: a GL-free wiring test (monkeypatches the MuJoCo
eval and the video writer, so it runs anywhere) and a guarded real-render test that
exercises the actual offscreen render path and skips when no GL context is available.
"""

from __future__ import annotations

import importlib
import math

import mujoco
import numpy as np
import pytest
import torch

pytest.importorskip("mujoco")
pytest.importorskip("so101_nexus.mujoco")


ppo_mod = importlib.import_module("examples.ppo_warp")

ENV_ID = "WarpTouch-v1"  # fastest task; no object spawn cost in the rollout
MUJOCO_ID = "MuJoCoTouch-v1"
EPISODE_LENGTH = 16
HIDDEN_DIM = 64


def _probe_dims():
    import gymnasium as gym

    import so101_nexus.mujoco  # noqa: F401 registers MuJoCo* envs

    env = gym.make(MUJOCO_ID, control_mode="pd_joint_delta_pos", max_episode_steps=EPISODE_LENGTH)
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape
    if obs_shape is None or act_shape is None:
        raise ValueError("rollout probe requires Box observation and action spaces")
    obs_dim = math.prod(obs_shape)
    act_dim = math.prod(act_shape)
    env.close()
    return obs_dim, act_dim


def _write_synthetic_checkpoint(path, obs_dim, act_dim):
    agent = ppo_mod.Agent(obs_dim, act_dim, HIDDEN_DIM)
    torch.save(
        {
            "model": agent.state_dict(),
            "obs_mean": torch.zeros(obs_dim, dtype=torch.float64),
            "obs_var": torch.ones(obs_dim, dtype=torch.float64),
            "step": 0,
            "success": 0.0,
        },
        path,
    )
    return agent


def test_rollout_video_from_checkpoint_wires_checkpoint(tmp_path, monkeypatch):
    """The helper loads the checkpoint, sizes the Agent from a probe env, and forwards
    the saved obs-norm stats into the MuJoCo eval + video writer without rendering."""
    obs_dim, act_dim = _probe_dims()
    ckpt_path = tmp_path / "best_agent.pt"
    expected_agent = _write_synthetic_checkpoint(ckpt_path, obs_dim, act_dim)
    expected_weight = expected_agent.actor_mean[0].weight.detach().cpu().clone()

    captured: dict = {}

    def fake_evaluate_mujoco(
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
        captured["kwargs"] = {
            "env_id": env_id,
            "control_mode": control_mode,
            "episode_length": episode_length,
            "eval_episodes": eval_episodes,
            "seed": seed,
            "capture_video": capture_video,
        }
        captured["obs_norm_mean"] = obs_norm.rms.mean.detach().cpu().clone()
        captured["obs_norm_var"] = obs_norm.rms.var.detach().cpu().clone()
        captured["agent_weight"] = agent.actor_mean[0].weight.detach().cpu().clone()
        return {
            "eval/return": 1.0,
            "eval/success_rate": 1.0,
            "eval/ep_len": float(EPISODE_LENGTH),
        }, [np.zeros((3, 3, 3), dtype=np.uint8)]

    def fake_write_video(frames, path, fps=30):
        captured["write_frames"] = frames
        captured["write_path"] = path
        return path

    monkeypatch.setattr(ppo_mod, "evaluate_mujoco", fake_evaluate_mujoco)
    monkeypatch.setattr(ppo_mod, "write_video", fake_write_video)

    out_path = tmp_path / "rollout.mp4"
    metrics, video_path = ppo_mod.rollout_video_from_checkpoint(
        str(ckpt_path),
        ENV_ID,
        control_mode="pd_joint_delta_pos",
        episode_length=EPISODE_LENGTH,
        hidden_dim=HIDDEN_DIM,
        seed=12345,
        out_path=str(out_path),
    )

    assert captured["kwargs"] == {
        "env_id": ENV_ID,
        "control_mode": "pd_joint_delta_pos",
        "episode_length": EPISODE_LENGTH,
        "eval_episodes": 1,
        "seed": 12345,
        "capture_video": True,
    }
    # Saved obs-norm stats and policy weights must load unchanged into the rollout.
    assert torch.allclose(captured["obs_norm_mean"], torch.zeros(obs_dim, dtype=torch.float64))
    assert torch.allclose(captured["obs_norm_var"], torch.ones(obs_dim, dtype=torch.float64))
    assert torch.allclose(captured["agent_weight"], expected_weight)
    assert set(metrics) >= {"eval/return", "eval/success_rate", "eval/ep_len"}
    assert video_path == str(out_path)
    assert captured["write_path"] == str(out_path)
    assert len(captured["write_frames"]) == 1


def test_rollout_video_from_checkpoint_renders_mp4(tmp_path):
    """End-to-end: a real offscreen MuJoCo render produces a non-empty mp4. Skips when
    no GL context (e.g. headless CI without EGL) is available."""
    obs_dim, act_dim = _probe_dims()
    ckpt_path = tmp_path / "best_agent.pt"
    _write_synthetic_checkpoint(ckpt_path, obs_dim, act_dim)

    out_path = tmp_path / "rollout.mp4"
    try:
        metrics, video_path = ppo_mod.rollout_video_from_checkpoint(
            str(ckpt_path),
            ENV_ID,
            control_mode="pd_joint_delta_pos",
            episode_length=EPISODE_LENGTH,
            hidden_dim=HIDDEN_DIM,
            seed=0,
            out_path=str(out_path),
        )
    except (mujoco.FatalError, RuntimeError) as exc:  # GL/render context unavailable
        msg = str(exc).lower()
        if any(k in msg for k in ("egl", "opengl", "gl ", "render", "context", "window")):
            pytest.skip(f"offscreen render unavailable in this environment: {exc}")
        raise

    assert video_path is not None
    assert out_path.is_file()
    assert out_path.stat().st_size > 0
    assert set(metrics) >= {"eval/return", "eval/success_rate", "eval/ep_len"}
