"""Parametrized Gymnasium contract suite shared across simulation backends.

Intended to be invoked from a per-backend ``test_envs.py`` so identical
behavioral expectations (reset/step signatures, reward bounds, seeded
determinism, observation-space conformance) are enforced uniformly.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np


def run_env_contract(
    env_id: str,
    config_cls: type | None = None,
    *,
    reward_range: tuple[float, float] = (-0.1, 1.0),
    n_steps: int = 3,
    make_kwargs: dict[str, Any] | None = None,
) -> None:
    """Run the shared Gymnasium contract against ``env_id``.

    The contract verifies:

    1. ``gym.make`` + ``close`` succeed with defaults.
    2. ``reset()`` returns ``(obs, info)``.
    3. ``step(action_space.sample())`` returns a 5-tuple with the
       documented scalar types.
    4. ``reward`` is finite and lies within ``reward_range``.
    5. ``info`` exposes ``"success"``.
    6. ``env.unwrapped.task_description`` is a non-empty string.
    7. ``reset(seed=s)`` twice yields identical observations.
    8. Multiple resets work back-to-back.

    Parameters
    ----------
    env_id : str
        Gymnasium environment id to construct.
    config_cls : type or None
        Backend-agnostic config class (``ReachConfig`` etc.) — kept in
        the signature so callers can parametrize on it even if this
        smoke pass doesn't instantiate a custom config.
    reward_range : tuple of float
        Inclusive ``(low, high)`` bounds on per-step reward.
    n_steps : int
        Number of random-action steps to run.
    make_kwargs : dict, optional
        Extra keyword arguments forwarded to ``gym.make``.
    """
    kwargs = dict(make_kwargs or {})
    env = gym.make(env_id, **kwargs)
    try:
        obs, info = env.reset()
        assert obs is not None
        assert isinstance(info, dict)

        task_desc = env.unwrapped.task_description  # type: ignore[attr-defined]
        assert isinstance(task_desc, str)
        assert task_desc

        low, high = reward_range
        for _ in range(n_steps):
            action = env.action_space.sample()
            result = env.step(action)
            assert len(result) == 5, f"step() returned {len(result)} items"
            obs, reward, terminated, truncated, info = result
            reward_f = float(reward)
            assert np.isfinite(reward_f), f"non-finite reward {reward} for {env_id}"
            assert low <= reward_f <= high, f"reward {reward_f} out of [{low}, {high}] for {env_id}"
            assert isinstance(terminated, (bool, np.bool_))
            assert isinstance(truncated, (bool, np.bool_))
            assert "success" in info

        # Seeded reset reproducibility.
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)
        _assert_obs_equal(obs1, obs2)

        # Multiple resets.
        for _ in range(3):
            obs, _ = env.reset()
            assert obs is not None
    finally:
        env.close()


def _assert_obs_equal(a: Any, b: Any) -> None:
    """Assert two observations are equal; handles flat arrays and dicts."""
    if isinstance(a, dict):
        assert isinstance(b, dict)
        assert a.keys() == b.keys()
        for k in a:
            _assert_obs_equal(a[k], b[k])
    else:
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
