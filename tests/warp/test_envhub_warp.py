"""LeRobot EnvHub contract for the Warp backend.

The Warp environments are natively batched and speak torch tensors, so the
EnvHub adapter has to present a NumPy vector env with per-world task strings.
These tests drive the same accessors LeRobot's rollout uses.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.warp


@pytest.fixture
def hub_env():
    """Build EnvHub Warp envs on the CPU device and close them after the test."""
    from so101_nexus.envhub import make_env

    built = []

    def _build(**kwargs):
        envs = make_env(device="cpu", **kwargs)
        env_id = next(iter(envs))
        built.append(envs[env_id][0])
        return env_id, built[-1]

    yield _build
    for env in built:
        env.close()


def test_batched_torch_observations_are_bridged_to_numpy(hub_env):
    _, env = hub_env(n_envs=3, env_id="WarpTouch-v1", episode_length=2)

    observation, _ = env.reset(seed=0)

    assert observation in env.observation_space
    assert isinstance(observation["agent_pos"], np.ndarray)
    assert observation["agent_pos"].shape == (3, 6)
    np.testing.assert_array_equal(observation["agent_pos"], observation["environment_state"][:, :6])


def test_worlds_expose_the_task_string_lerobot_reads(hub_env):
    lerobot_utils = pytest.importorskip("lerobot.envs.utils")
    _, env = hub_env(n_envs=2, env_id="WarpTouch-v1", episode_length=2)
    observation, _ = env.reset(seed=0)

    assert hasattr(env.envs[0], "task_description")
    assert hasattr(env.envs[0], "task")

    features = lerobot_utils.preprocess_observation(observation)
    tasks = lerobot_utils.add_envs_task(env, features)["task"]

    assert tasks == list(env.unwrapped.task_descriptions)
    assert tasks[0].startswith("Touch the ")


def test_step_takes_numpy_actions_and_reports_success_in_final_info(hub_env):
    _, env = hub_env(n_envs=2, env_id="WarpTouch-v1", episode_length=2)
    env.reset(seed=0)
    action = np.zeros((2, 6), dtype=np.float32)

    _, reward, terminated, truncated, info = env.step(action)
    assert isinstance(reward, np.ndarray)
    assert terminated.dtype == np.bool_
    assert "final_info" not in info

    _, _, _, truncated, info = env.step(action)
    assert truncated.all()
    np.testing.assert_array_equal(
        info["final_info"]["is_success"], np.asarray(info["success"], dtype=bool)
    )


def test_max_episode_steps_is_reported_per_world(hub_env):
    _, env = hub_env(n_envs=3, env_id="WarpMove-v1", episode_length=5)

    assert env.call("_max_episode_steps") == (5, 5, 5)


def test_a_per_world_seed_list_collapses_to_one_batch_seed(hub_env):
    _, env = hub_env(n_envs=2, env_id="WarpTouch-v1", episode_length=2)

    listed, _ = env.reset(seed=[3, 4])
    scalar, _ = env.reset(seed=3)

    # Same episode, not a bitwise replay: the CPU solver's settle is not exactly
    # reproducible across resets of a live env.
    np.testing.assert_allclose(listed["environment_state"], scalar["environment_state"], atol=1e-6)


def test_pixels_obs_type_emits_numpy_camera_batches(hub_env):
    _, env = hub_env(
        n_envs=2,
        env_id="WarpTouch-v1",
        obs_type="pixels_agent_pos",
        observation_width=24,
        observation_height=16,
        episode_length=2,
    )

    observation, _ = env.reset(seed=0)

    assert set(observation) == {"agent_pos", "pixels"}
    assert set(observation["pixels"]) == {"wrist", "overhead"}
    for image in observation["pixels"].values():
        assert image.shape == (2, 16, 24, 3)
        assert image.dtype == np.uint8
