"""Golden-value determinism regression tests for the MuJoCo backend.

These tests pin the exact numeric behavior of every MuJoCo environment for a
fixed seed to a committed fixture (``data/determinism_goldens.npz``). They exist
to catch silent physics or reward drift: a dependency bump (MuJoCo, NumPy) or a
code change that quietly alters simulated dynamics, observation assembly, or a
reward formula will move these numbers and redden the corresponding test, even
when every shape/space invariant still holds.

Two complementary checks are provided:

* Golden (cross-version): a fresh rollout must match the committed fixture within
  ``rtol=1e-5, atol=1e-6``. The tolerance (rather than bit-exact equality) is
  deliberate: it still catches any real drift (physics/reward changes move values
  far more than a last-bit rounding difference) while tolerating harmless
  last-bit differences that a different CPU/BLAS/compiler build may introduce.
* Intra-run (same process): two independent same-seed rollouts must be bit-for-bit
  identical (``assert_array_equal``). This is a strict, machine-local guarantee
  that seeding is complete and no hidden global state leaks between rollouts.

GL independence: every MuJoCo env here defaults to ``obs_mode="state"`` with no
camera observation component, so the observation is a flat proprioceptive/state
vector (a ``gymnasium.spaces.Box``) built without any rendering. That keeps the
goldens portable across GL backends and CI machines. ``_rollout`` asserts the
observation space is a ``Box`` so a future default that silently adds a pixel
component (which would make goldens GL-dependent) fails loudly instead of
committing non-portable numbers.

Observation flattening: observations are flattened to a 1-D array with
``_flatten_obs``. State-only envs already return a flat array; the helper also
handles dict observations by concatenating components in sorted-key order, giving
a deterministic layout independent of dict insertion order.

Regeneration: goldens are generated from the true env behavior, never fabricated.
To rewrite the fixture from current runtime values (for example after an
intentional, reviewed physics/reward change), run::

    REGEN_GOLDENS=1 uv run pytest tests/mujoco/test_determinism_goldens.py

Review the resulting diff to ``data/determinism_goldens.npz`` before committing.
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

import so101_nexus.mujoco  # noqa: F401 - registers the MuJoCo*-v1 env IDs

ENV_IDS = [
    "MuJoCoTouch-v1",
    "MuJoCoLookAt-v1",
    "MuJoCoMove-v1",
    "MuJoCoPickLift-v1",
    "MuJoCoPickAndPlace-v1",
]

SEED = 0
N_STEPS = 10
GOLDEN_PATH = Path(__file__).parent / "data" / "determinism_goldens.npz"

RTOL = 1e-5
ATOL = 1e-6


def _flatten_obs(obs: object) -> np.ndarray:
    """Flatten an observation to a 1-D array with a deterministic layout.

    State-only envs return a flat array already. Dict observations are
    concatenated in sorted-key order so the layout does not depend on dict
    insertion order.
    """
    if isinstance(obs, dict):
        return np.concatenate([np.asarray(obs[key]).ravel() for key in sorted(obs)])
    return np.asarray(obs).ravel()


def _rollout(env_id: str, seed: int, n_steps: int) -> dict[str, np.ndarray]:
    """Run a deterministic seeded rollout and capture its numeric fingerprint.

    Returns the flattened reset observation, the flattened final observation, the
    per-step reward sequence, and the ``[terminated, truncated]`` flags at the end
    of a fixed ``n_steps`` rollout driven by seeded ``action_space.sample()``.
    """
    env = gym.make(env_id)
    try:
        assert isinstance(env.observation_space, spaces.Box), (
            f"{env_id} observation space is {type(env.observation_space).__name__}, "
            "expected Box: determinism goldens require a state-only (no camera) obs "
            "config so they stay GL-independent"
        )
        obs, _ = env.reset(seed=seed)
        reset_obs = _flatten_obs(obs)

        env.action_space.seed(seed)
        rewards: list[float] = []
        terminated = False
        truncated = False
        final_obs = reset_obs
        for _ in range(n_steps):
            obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
            rewards.append(float(reward))
            final_obs = _flatten_obs(obs)

        return {
            "reset_obs": reset_obs,
            "final_obs": final_obs,
            "rewards": np.asarray(rewards, dtype=np.float64),
            "flags": np.asarray([terminated, truncated], dtype=bool),
        }
    finally:
        env.close()


def _regenerate_goldens() -> None:
    data: dict[str, np.ndarray] = {}
    for env_id in ENV_IDS:
        rollout = _rollout(env_id, SEED, N_STEPS)
        for key, value in rollout.items():
            data[f"{env_id}/{key}"] = value
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(GOLDEN_PATH, **data)


@pytest.fixture(scope="module")
def goldens():
    if os.environ.get("REGEN_GOLDENS") == "1":
        _regenerate_goldens()
    if not GOLDEN_PATH.exists():
        raise FileNotFoundError(
            f"Golden fixture {GOLDEN_PATH} is missing; regenerate with "
            "REGEN_GOLDENS=1 uv run pytest tests/mujoco/test_determinism_goldens.py"
        )
    with np.load(GOLDEN_PATH) as loaded:
        yield {key: loaded[key] for key in loaded.files}


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_matches_golden(env_id, goldens):
    """A fresh seeded rollout matches the committed golden fingerprint."""
    rollout = _rollout(env_id, SEED, N_STEPS)

    np.testing.assert_allclose(
        rollout["reset_obs"],
        goldens[f"{env_id}/reset_obs"],
        rtol=RTOL,
        atol=ATOL,
        err_msg=f"{env_id} reset observation drifted from golden",
    )
    np.testing.assert_allclose(
        rollout["final_obs"],
        goldens[f"{env_id}/final_obs"],
        rtol=RTOL,
        atol=ATOL,
        err_msg=f"{env_id} final observation drifted from golden",
    )
    np.testing.assert_allclose(
        rollout["rewards"],
        goldens[f"{env_id}/rewards"],
        rtol=RTOL,
        atol=ATOL,
        err_msg=f"{env_id} reward sequence drifted from golden",
    )
    np.testing.assert_array_equal(
        rollout["flags"],
        goldens[f"{env_id}/flags"],
        err_msg=f"{env_id} terminated/truncated flags drifted from golden",
    )


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_intra_run_determinism(env_id):
    """Two same-seed rollouts in one process are bit-for-bit identical."""
    first = _rollout(env_id, SEED, N_STEPS)
    second = _rollout(env_id, SEED, N_STEPS)

    np.testing.assert_array_equal(
        first["reset_obs"], second["reset_obs"], err_msg=f"{env_id} reset obs not deterministic"
    )
    np.testing.assert_array_equal(
        first["final_obs"], second["final_obs"], err_msg=f"{env_id} final obs not deterministic"
    )
    np.testing.assert_array_equal(
        first["rewards"], second["rewards"], err_msg=f"{env_id} rewards not deterministic"
    )
    np.testing.assert_array_equal(
        first["flags"], second["flags"], err_msg=f"{env_id} flags not deterministic"
    )


if __name__ == "__main__":
    _regenerate_goldens()
    print(f"Wrote goldens for {len(ENV_IDS)} envs to {GOLDEN_PATH}")
