"""LeRobot EnvHub entry-point contract for the MuJoCo environments.

The Hub loads a repository file and calls its ``make_env``, so these tests drive
the same path LeRobot does: ``lerobot.envs.utils`` loaders on the committed
``envhub/`` files, and ``preprocess_observation`` / ``add_envs_task`` on what
comes back.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

import so101_nexus.mujoco  # registers MuJoCo*-v1
import so101_nexus.warp  # noqa: F401  (registers Warp*-v1)
from so101_nexus import JointPositions, PickConfig, component_slice
from so101_nexus.env_ids import all_registered_env_ids
from so101_nexus.envhub import DEFAULT_ENV_ID, make_env

lerobot_utils = pytest.importorskip("lerobot.envs.utils")

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
        env.close()


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


def test_pixels_obs_type_emits_cameras_and_withholds_privileged_state(hub_env):
    _, env = hub_env(
        n_envs=1,
        env_id="MuJoCoTouch-v1",
        obs_type="pixels_agent_pos",
        observation_width=32,
        observation_height=24,
        episode_length=2,
    )
    observation, _ = env.reset(seed=0)

    assert set(observation) == {"agent_pos", "pixels"}
    assert set(observation["pixels"]) == {"wrist", "overhead"}
    for image in observation["pixels"].values():
        assert image.shape == (1, 24, 32, 3)
        assert image.dtype == np.uint8


def test_observations_survive_lerobot_preprocessing(hub_env):
    _, env = hub_env(
        n_envs=2,
        env_id="MuJoCoTouch-v1",
        obs_type="pixels_agent_pos",
        observation_width=32,
        observation_height=24,
        episode_length=2,
    )
    observation, _ = env.reset(seed=0)

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

    _, _, _, _, info = env.step(env.action_space.sample())
    np.testing.assert_array_equal(info["is_success"], info["success"])

    _, _, _, truncated, info = env.step(env.action_space.sample())
    assert truncated.all()
    np.testing.assert_array_equal(info["final_info"]["is_success"], info["success"])


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
    _, env = hub_env(n_envs=1, env_id="MuJoCoPickLift-v1", config=config, episode_length=2)
    observation, _ = env.reset(seed=0)

    assert observation["environment_state"].shape == (1, 6)


def test_config_object_fields_select_the_environment(hub_env):
    class Cfg:
        task = "MuJoCoLookAt-v1"
        episode_length = 3
        kwargs = {"control_mode": "pd_joint_delta_pos"}

    env_id, env = hub_env(n_envs=1, cfg=Cfg())

    assert env_id == "MuJoCoLookAt-v1"
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

    assert set(envs) == {DEFAULT_ENV_ID}
    envs[DEFAULT_ENV_ID][0].close()
