"""LeRobot EnvHub entry-point contract for the MuJoCo environments.

The Hub loads a repository file and calls its ``make_env``, so these tests drive
the same path LeRobot does: ``lerobot.envs.utils`` loaders on the committed
``envhub/`` files, and ``preprocess_observation`` / ``add_envs_task`` on what
comes back.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
import pytest

import so101_nexus.mujoco  # registers MuJoCo*-v1
import so101_nexus.warp  # noqa: F401  (registers Warp*-v1)
from so101_nexus import (
    JointPositions,
    JointVelocities,
    PickConfig,
    StackCubeConfig,
    TouchConfig,
    component_slice,
    privileged_state_feature_names,
)
from so101_nexus.env_ids import all_registered_env_ids
from so101_nexus.envhub import DEFAULT_ENV_ID, EnvHubAdapter, make_env

lerobot_utils = pytest.importorskip("lerobot.envs.utils")
HubEnvConfig = pytest.importorskip("lerobot.envs.configs").HubEnvConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
ENVHUB_DIR = REPO_ROOT / "envhub"


@pytest.fixture
def hub_env():
    """Build EnvHub vector envs and close them after the test."""
    built = []

    def _build(**kwargs):
        envs = make_env(**kwargs)
        env_id = next(iter(envs))
        built.append(envs[env_id][0])
        return env_id, built[-1]

    yield _build
    for env in built:
        with contextlib.suppress(Exception):
            env.close()


def _pixels_env_or_skip(hub_env, **kwargs):
    """Build a camera env, skipping when this machine has no GL context.

    Camera observations are rendered during ``reset``, so the skip that
    ``tests/mujoco/test_render_camera.py`` applies to explicit ``render()`` calls
    has to wrap the reset here.
    """
    _, env = hub_env(obs_type="pixels_agent_pos", **kwargs)
    try:
        return env, env.reset(seed=0)[0]
    except (mujoco.FatalError, RuntimeError) as exc:
        msg = str(exc).lower()
        if any(k in msg for k in ("egl", "opengl", "gl ", "render", "context", "window")):
            pytest.skip(f"offscreen render unavailable in this environment: {exc}")
        raise


class _SucceedingEnv(gym.Env):
    """Minimal SO101-shaped env that terminates reporting ``info["success"]``."""

    metadata = {"render_modes": []}
    task_description = "Succeed immediately."

    class _Config:
        observations = [JointPositions()]

    def __init__(self) -> None:
        super().__init__()
        self.config = self._Config()
        self.observation_space = gym.spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):  # type: ignore[override]
        super().reset(seed=seed)
        return np.zeros(6, dtype=np.float32), {"success": False}

    def step(self, action):  # type: ignore[override]
        return np.zeros(6, dtype=np.float32), 1.0, True, False, {"success": True}


def test_make_env_returns_the_hub_result_mapping(hub_env):
    env_id, env = hub_env(n_envs=2, env_id="MuJoCoTouch-v1", episode_length=2)

    assert env_id == "MuJoCoTouch-v1"
    assert isinstance(env, gym.vector.VectorEnv)
    assert env.num_envs == 2


def test_default_env_id_is_served_without_a_task(hub_env):
    env_id, _ = hub_env(n_envs=1, episode_length=2)

    assert env_id == DEFAULT_ENV_ID


def test_state_observation_splits_joint_positions_from_the_state_vector(hub_env):
    _, env = hub_env(n_envs=2, env_id="MuJoCoTouch-v1", episode_length=2)
    observation, _ = env.reset(seed=0)

    assert observation in env.observation_space
    joints = component_slice(env.envs[0].unwrapped.config.observations, JointPositions)
    np.testing.assert_array_equal(
        observation["agent_pos"], observation["environment_state"][:, joints]
    )


def test_agent_pos_follows_the_component_layout_off_offset_zero(hub_env):
    """Joint positions are located by component, not assumed to lead the vector."""
    config = TouchConfig(observations=[JointVelocities(), JointPositions()])
    _, env = hub_env(n_envs=1, env_id="MuJoCoTouch-v1", config=config, episode_length=2)
    observation, _ = env.reset(seed=0)

    assert observation["environment_state"].shape == (1, 12)
    np.testing.assert_array_equal(
        observation["agent_pos"], observation["environment_state"][:, 6:12]
    )


def test_pixels_obs_type_emits_cameras_and_withholds_privileged_state(hub_env):
    env, observation = _pixels_env_or_skip(
        hub_env,
        n_envs=1,
        env_id="MuJoCoTouch-v1",
        observation_width=32,
        observation_height=24,
        episode_length=2,
    )

    assert observation in env.observation_space
    assert set(observation) == {"agent_pos", "pixels"}
    assert observation["agent_pos"].shape == (1, 6)
    assert set(observation["pixels"]) == {"wrist", "overhead"}
    for image in observation["pixels"].values():
        assert image.shape == (1, 24, 32, 3)
        assert image.dtype == np.uint8


def test_pixels_obs_type_keeps_the_task_privileged_state(hub_env):
    """The camera config must not shrink the asymmetric actor-critic's ground truth.

    ``obs_mode="visual"`` routes the non-camera components to
    ``info["privileged_state"]``. Rebuilding the task with joint positions alone
    would collapse that channel to a copy of ``agent_pos``.
    """
    _, info = _pixels_env_or_skip(
        hub_env,
        n_envs=1,
        env_id="MuJoCoStackCube-v1",
        observation_width=16,
        observation_height=16,
        episode_length=2,
    )[0].reset(seed=0)

    expected = len(privileged_state_feature_names(StackCubeConfig().observations))
    assert expected > 6
    assert info["privileged_state"].shape == (1, expected)


def test_observations_survive_lerobot_preprocessing(hub_env):
    env, observation = _pixels_env_or_skip(
        hub_env,
        n_envs=2,
        env_id="MuJoCoTouch-v1",
        observation_width=32,
        observation_height=24,
        episode_length=2,
    )

    features = lerobot_utils.preprocess_observation(observation)

    assert set(features) == {
        "observation.state",
        "observation.images.wrist",
        "observation.images.overhead",
    }
    assert features["observation.images.wrist"].shape == (2, 3, 24, 32)


def test_task_description_reaches_lerobot(hub_env):
    _, env = hub_env(n_envs=2, env_id="MuJoCoTouch-v1", episode_length=2)
    observation, _ = env.reset(seed=0)

    # The exact attribute probe LeRobot runs before reading the instruction.
    assert hasattr(env.envs[0], "task_description")
    assert hasattr(env.envs[0], "task")

    features = lerobot_utils.preprocess_observation(observation)
    tasks = lerobot_utils.add_envs_task(env, features)["task"]

    assert tasks == [env.envs[0].task_description] * 2
    assert tasks[0].startswith("Touch the ")


def test_success_is_mirrored_to_the_key_lerobot_reads(hub_env):
    _, env = hub_env(n_envs=2, env_id="MuJoCoTouch-v1", episode_length=2)
    env.reset(seed=0)
    env.action_space.seed(0)

    _, _, _, _, info = env.step(env.action_space.sample())
    np.testing.assert_array_equal(info["is_success"], info["success"])

    _, _, _, truncated, info = env.step(env.action_space.sample())
    assert truncated.all()
    # Same-step autoreset makes the top-level info the next episode's; the
    # terminated episode's flags are the ones LeRobot reads out of final_info.
    np.testing.assert_array_equal(info["final_info"]["is_success"], info["final_info"]["success"])


def test_a_successful_episode_reports_is_success_true():
    """The True case: a rollout scores success out of ``final_info``, not ``success``."""
    env = gym.vector.SyncVectorEnv(
        [lambda: EnvHubAdapter(_SucceedingEnv())] * 2,
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    try:
        env.reset(seed=0)
        _, _, terminated, _, info = env.step(np.zeros((2, 6), dtype=np.float32))

        assert terminated.all()
        assert info["final_info"]["is_success"].tolist() == [True, True]
    finally:
        env.close()


def test_episode_length_and_control_mode_reach_the_environment(hub_env):
    _, env = hub_env(
        n_envs=2,
        env_id="MuJoCoTouch-v1",
        episode_length=7,
        control_mode="pd_ee_pose",
    )

    assert env.call("_max_episode_steps") == (7, 7)
    assert env.action_space.shape == (2, 7)


def test_explicit_config_overrides_the_obs_type_defaults(hub_env):
    config = PickConfig(observations=[JointPositions()])
    _, env = hub_env(
        n_envs=1,
        env_id="MuJoCoPickLift-v1",
        config=config,
        obs_type="pixels_agent_pos",
        episode_length=2,
    )
    observation, _ = env.reset(seed=0)

    # config wins over obs_type: no cameras were built, so there are no pixels.
    assert set(observation) == {"agent_pos", "environment_state"}
    assert observation["environment_state"].shape == (1, 6)


def test_lerobot_env_config_selects_the_environment(hub_env):
    cfg = HubEnvConfig(hub_path="johnsutor/so101-nexus-envs", task="MuJoCoLookAt-v1")

    env_id, env = hub_env(n_envs=1, cfg=cfg, episode_length=3)

    assert env_id == "MuJoCoLookAt-v1"
    assert env.call("_max_episode_steps") == (3,)


def test_config_kwargs_are_applied_and_keyword_overrides_win(hub_env):
    """``cfg.kwargs`` reaches the env, and a keyword override beats a cfg field.

    The per-environment Hub shims rely on the second half: they bind ``env_id``
    as a keyword override, which must win over a config's ``task``.
    """

    class Cfg:
        task = "MuJoCoLookAt-v1"
        episode_length = 3
        kwargs = {"control_mode": "pd_ee_pose"}

    env_id, env = hub_env(n_envs=1, cfg=Cfg(), env_id="MuJoCoTouch-v1")

    assert env_id == "MuJoCoTouch-v1"
    assert env.action_space.shape == (1, 7)
    assert env.call("_max_episode_steps") == (3,)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"n_envs": 0}, "n_envs must be >= 1"),
        ({"obs_type": "pixels"}, "obs_type must be"),
        ({"observation_depth": 3}, "Unknown make_env options"),
        ({"env_id": "PandaPick-v1"}, "known backend prefix"),
    ],
)
def test_invalid_requests_are_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        make_env(**{"n_envs": 1, "episode_length": 2, **kwargs})


def test_async_vectorization_is_selectable(hub_env):
    _, env = hub_env(n_envs=2, env_id="MuJoCoTouch-v1", episode_length=2, use_async_envs=True)
    observation, _ = env.reset(seed=0)

    assert isinstance(env, gym.vector.AsyncVectorEnv)
    assert observation["agent_pos"].shape == (2, 6)


def test_hub_package_ships_one_entry_point_per_registered_environment():
    """Every registered id is loadable from the Hub, and nothing else is."""
    shipped = {path.stem for path in (ENVHUB_DIR / "envs").glob("*.py")}

    assert shipped == set(all_registered_env_ids())


@pytest.mark.parametrize("env_id", sorted(all_registered_env_ids()))
def test_every_hub_entry_point_binds_its_environment_id(env_id):
    module = lerobot_utils._load_module_from_path(str(ENVHUB_DIR / "envs" / f"{env_id}.py"))

    assert module.make_env.keywords == {"env_id": env_id}


def test_hub_entry_point_runs_through_the_lerobot_loader():
    """The published file, loaded and called exactly as LeRobot's EnvHub does."""
    module = lerobot_utils._load_module_from_path(str(ENVHUB_DIR / "envs" / "MuJoCoTouch-v1.py"))

    result = lerobot_utils._call_make_env(module, n_envs=2, use_async_envs=False, cfg=None)
    envs = lerobot_utils._normalize_hub_result(result)

    env = envs["MuJoCoTouch-v1"][0]
    try:
        observation, _ = env.reset(seed=0)
        assert observation["agent_pos"].shape == (2, 6)
    finally:
        env.close()


def test_root_hub_entry_point_serves_the_default_environment():
    module = lerobot_utils._load_module_from_path(str(ENVHUB_DIR / "env.py"))

    envs = lerobot_utils._normalize_hub_result(
        lerobot_utils._call_make_env(module, n_envs=1, use_async_envs=False, cfg=None)
    )

    try:
        assert set(envs) == {DEFAULT_ENV_ID}
    finally:
        lerobot_utils.close_envs(envs)


def test_root_hub_entry_point_honors_a_config_task():
    """The documented flow: root ``env.py`` plus a config whose ``task`` picks the env."""
    module = lerobot_utils._load_module_from_path(str(ENVHUB_DIR / "env.py"))
    cfg = HubEnvConfig(hub_path="johnsutor/so101-nexus-envs", task="MuJoCoMove-v1")

    envs = lerobot_utils._normalize_hub_result(
        lerobot_utils._call_make_env(module, n_envs=1, use_async_envs=False, cfg=cfg)
    )

    try:
        assert set(envs) == {"MuJoCoMove-v1"}
    finally:
        lerobot_utils.close_envs(envs)
