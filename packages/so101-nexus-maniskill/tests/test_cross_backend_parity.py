"""Cross-backend guardrails comparing the MuJoCo and ManiSkill SO101 envs.

This module imports BOTH backends (it lives in the ManiSkill test tree, which is
permitted to import MuJoCo). It builds one single-env (num_envs=1, CPU) Reach
environment per backend for every shared ``ControlMode`` and asserts the public
action-space contract is identical: same shape and same low/high bounds.

After the delta-action normalization both delta modes expose a [-1, 1] box, and
``pd_joint_pos`` exposes the shared joint-limit-clamped target range, so the two
backends must agree exactly. It also confirms episode length is owned by the gym
registration (``env.spec.max_episode_steps``) rather than the config object.
"""

from __future__ import annotations

import typing

import gymnasium as gym
import numpy as np
import pytest

import so101_nexus_maniskill  # noqa: F401 - registers ManiSkill envs

# This parity module needs both backends installed. The ManiSkill CI job does
# not install the MuJoCo package, so skip the whole module when it is absent.
so101_nexus_mujoco = pytest.importorskip(
    "so101_nexus_mujoco"
)  # noqa: F841 - registers MuJoCo envs
from so101_nexus_core.config import ReachConfig
from so101_nexus_core.testing import skip_if_vectorized_runtime_unavailable


def _control_modes() -> list[str]:
    """Resolve the shared ControlMode literal values to a concrete list."""
    from so101_nexus_core.config import ControlMode

    modes = list(typing.get_args(ControlMode))
    assert modes, "ControlMode literal yielded no values"
    return modes


CONTROL_MODES = _control_modes()


def _single_action_space(env):
    """Return the per-env (unbatched) action space for either backend.

    ManiSkill vector envs expose ``single_action_space``; the MuJoCo env exposes
    a plain (unbatched) ``action_space``.
    """
    inner = env.unwrapped
    return getattr(inner, "single_action_space", inner.action_space)


@pytest.mark.parametrize("control_mode", CONTROL_MODES)
def test_reach_action_space_parity_across_backends(control_mode):
    """MuJoCo and ManiSkill SO101 Reach expose the same action-space contract."""
    mj_env = gym.make("MuJoCoReach-v1", config=ReachConfig(), control_mode=control_mode)
    try:
        ms_env = gym.make(
            "ManiSkillReachSO101-v1",
            config=ReachConfig(),
            num_envs=1,
            obs_mode="state",
            render_mode=None,
            control_mode=control_mode,
        )
    except Exception as exc:  # narrowed: only GPU-availability errors become skips
        mj_env.close()
        skip_if_vectorized_runtime_unavailable(exc)

    try:
        mj_space = _single_action_space(mj_env)
        ms_space = _single_action_space(ms_env)

        assert tuple(mj_space.shape) == tuple(ms_space.shape), (
            f"action-space shape diverged for {control_mode!r}: "
            f"mujoco={mj_space.shape} maniskill={ms_space.shape}"
        )

        mj_low = np.asarray(mj_space.low, dtype=np.float32)
        mj_high = np.asarray(mj_space.high, dtype=np.float32)
        ms_low = np.asarray(ms_space.low, dtype=np.float32)
        ms_high = np.asarray(ms_space.high, dtype=np.float32)

        if control_mode == "pd_joint_pos":
            # Documented contract: position targets span the shared joint-limit-
            # clamped range (not normalized).
            np.testing.assert_allclose(mj_low, ms_low, rtol=0, atol=1e-4)
            np.testing.assert_allclose(mj_high, ms_high, rtol=0, atol=1e-4)
        else:
            # Documented contract: both delta modes expose a normalized [-1, 1]
            # action space on both backends.
            np.testing.assert_allclose(mj_low, -np.ones_like(mj_low), atol=1e-6)
            np.testing.assert_allclose(mj_high, np.ones_like(mj_high), atol=1e-6)
            np.testing.assert_allclose(ms_low, -np.ones_like(ms_low), atol=1e-6)
            np.testing.assert_allclose(ms_high, np.ones_like(ms_high), atol=1e-6)
            np.testing.assert_allclose(mj_low, ms_low, atol=1e-6)
            np.testing.assert_allclose(mj_high, ms_high, atol=1e-6)
    finally:
        mj_env.close()
        ms_env.close()


def test_mujoco_episode_length_owned_by_gym_spec():
    """The MuJoCo Reach episode limit comes from the gym spec, not config."""
    config = ReachConfig()
    assert not hasattr(config, "max_episode_steps")

    env = gym.make("MuJoCoReach-v1", config=config)
    try:
        assert env.spec is not None
        assert env.spec.max_episode_steps is not None
        assert env.spec.max_episode_steps > 0
    finally:
        env.close()


def test_maniskill_episode_length_owned_by_registration():
    """The ManiSkill episode limit comes from register_env, not config.

    Documented backend divergence: MuJoCo carries max_episode_steps on the
    Gymnasium EnvSpec, but ManiSkill stores it on its own env registry (set via
    ``register_env(..., max_episode_steps=...)``) and applies it through a
    TimeLimit wrapper, leaving ``env.spec.max_episode_steps`` as None. We assert
    the documented contract for this backend rather than skipping.
    """
    from mani_skill.utils.registration import REGISTERED_ENVS

    config = ReachConfig()
    assert not hasattr(config, "max_episode_steps")

    try:
        env = gym.make(
            "ManiSkillReachSO101-v1",
            config=config,
            num_envs=1,
            obs_mode="state",
            render_mode=None,
        )
    except Exception as exc:  # narrowed: only GPU-availability errors become skips
        skip_if_vectorized_runtime_unavailable(exc)
    try:
        spec = REGISTERED_ENVS.get("ManiSkillReachSO101-v1")
        assert spec is not None, "ManiSkillReachSO101-v1 is not registered"
        assert spec.max_episode_steps is not None
        assert spec.max_episode_steps > 0
    finally:
        env.close()
