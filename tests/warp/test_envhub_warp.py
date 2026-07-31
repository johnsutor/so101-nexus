"""LeRobot EnvHub contract for the Warp backend.

The Warp environments are natively batched and speak torch tensors, so the
EnvHub adapter has to present a NumPy vector env with per-world task strings.
These tests drive the same accessors LeRobot's rollout uses.
"""

import contextlib

import gymnasium as gym
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
        with contextlib.suppress(Exception):
            env.close()


@pytest.fixture
def scripted_env():
    """Wrap a batched stand-in whose per-world done and success flags are scripted.

    A real Warp batch cannot be driven to a partially terminated step cheaply, and
    that is the only state where the adapter's ``success & done`` mask does any
    work, so the batched contract is pinned against a stub instead.
    """
    import torch

    from so101_nexus.observations import JointPositions

    class _Config:
        observations = [JointPositions()]

    class _FakeWarpEnv(gym.vector.VectorEnv):
        """Three worlds: world 0 succeeds and ends, world 1 succeeds without ending."""

        def __init__(self) -> None:
            self.num_envs = 3
            self.device = torch.device("cpu")
            self.config = _Config()
            self.max_episode_steps = 4
            self.task_descriptions = ["Do the thing."] * self.num_envs
            self.single_observation_space = gym.spaces.Box(-1.0, 1.0, (6,), np.float32)
            self.single_action_space = gym.spaces.Box(-1.0, 1.0, (6,), np.float32)

        def reset(self, *, seed=None, options=None):
            return torch.zeros(self.num_envs, 6), {}

        def step(self, actions):
            assert isinstance(actions, torch.Tensor)
            return (
                torch.zeros(self.num_envs, 6),
                torch.ones(self.num_envs),
                torch.tensor([True, False, False]),
                torch.tensor([False, False, True]),
                {"success": torch.tensor([True, True, False])},
            )

    from so101_nexus.envhub import WarpEnvHubAdapter

    return WarpEnvHubAdapter(_FakeWarpEnv())


def test_batched_torch_observations_are_bridged_to_numpy(hub_env):
    from so101_nexus.observations import JointPositions, component_slice

    _, env = hub_env(n_envs=3, env_id="WarpTouch-v1", episode_length=2)

    observation, _ = env.reset(seed=0)

    assert observation in env.observation_space
    assert isinstance(observation["agent_pos"], np.ndarray)
    assert observation["agent_pos"].shape == (3, 6)
    joints = component_slice(env.unwrapped.config.observations, JointPositions)
    np.testing.assert_array_equal(
        observation["agent_pos"], observation["environment_state"][:, joints]
    )


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


def test_only_worlds_that_ended_report_success(scripted_env):
    """``final_info["is_success"]`` is success masked by done, per world.

    World 1 succeeds without terminating: reporting it would double-count the
    episode LeRobot has not seen end yet.
    """
    scripted_env.reset(seed=0)

    _, _, terminated, truncated, info = scripted_env.step(np.zeros((3, 6), dtype=np.float32))

    assert terminated.tolist() == [True, False, False]
    assert truncated.tolist() == [False, False, True]
    assert info["final_info"]["is_success"].tolist() == [True, False, False]


def test_call_splits_per_world_values_and_repeats_scalars(scripted_env):
    assert scripted_env.call("max_episode_steps") == (4, 4, 4)
    assert scripted_env.call("task_descriptions") == ("Do the thing.",) * 3


def test_max_episode_steps_is_reported_per_world(hub_env):
    _, env = hub_env(n_envs=3, env_id="WarpMove-v1", episode_length=5)

    assert env.call("_max_episode_steps") == (5, 5, 5)


def test_a_per_world_seed_list_collapses_to_one_batch_seed(hub_env):
    _, env = hub_env(n_envs=2, env_id="WarpTouch-v1", episode_length=2)

    listed, _ = env.reset(seed=[3, 4])
    scalar, _ = env.reset(seed=3)
    other, _ = env.reset(seed=4)

    # Same episode, not a bitwise replay: the CPU solver's settle is not exactly
    # reproducible across resets of a live env. Seed 4 is the negative control.
    np.testing.assert_allclose(listed["environment_state"], scalar["environment_state"], atol=1e-8)
    assert not np.allclose(listed["environment_state"], other["environment_state"], atol=1e-8)


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
    assert observation["agent_pos"].shape == (2, 6)
    assert set(observation["pixels"]) == {"wrist", "overhead"}
    for image in observation["pixels"].values():
        assert image.shape == (2, 16, 24, 3)
        assert image.dtype == np.uint8
