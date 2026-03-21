"""Tests for the unified MuJoCo PickEnv / PickLiftEnv."""

import gymnasium as gym
import pytest

import so101_nexus_mujoco  # noqa: F401
from so101_nexus_core import CubeObject, PickConfig, YCBObject


@pytest.fixture(scope="module")
def pick_env():
    env = gym.make("MuJoCoPickLift-v1")
    yield env
    env.close()


class TestPickEnvDefaults:
    def test_make(self, pick_env):
        assert pick_env is not None

    def test_obs_shape(self, pick_env):
        obs, _ = pick_env.reset()
        assert obs.shape == (18,)  # tcp_pose(7) + is_grasped(1) + obj_pose(7) + tcp_to_obj(3)

    def test_step_returns_five_tuple(self, pick_env):
        pick_env.reset()
        assert len(pick_env.step(pick_env.action_space.sample())) == 5

    def test_task_description_contains_repr(self, pick_env):
        pick_env.reset()
        assert "red cube" in pick_env.unwrapped.task_description

    def test_info_has_success(self, pick_env):
        pick_env.reset()
        _, _, _, _, info = pick_env.step(pick_env.action_space.sample())
        assert "success" in info
        assert "lift_height" in info


class TestPickEnvObjectConfig:
    def test_ycb_object(self):
        env = gym.make(
            "MuJoCoPickLift-v1", config=PickConfig(objects=[YCBObject(model_id="009_gelatin_box")])
        )
        obs, _ = env.reset()
        assert obs.shape == (18,)
        assert "gelatin box" in env.unwrapped.task_description
        env.close()

    def test_multiple_objects_with_distractors(self):
        env = gym.make(
            "MuJoCoPickLift-v1",
            config=PickConfig(
                objects=[
                    CubeObject(color="red"),
                    CubeObject(color="blue"),
                    CubeObject(color="green"),
                ],
                n_distractors=2,
            ),
        )
        obs, _ = env.reset()
        assert obs.shape == (18,)
        env.close()
