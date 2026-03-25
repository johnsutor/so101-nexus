"""Tests for MuJoCoLookAt-v1."""

import gymnasium as gym
import numpy as np
import pytest

import so101_nexus_mujoco  # noqa: F401


@pytest.fixture(scope="module")
def look_at_env():
    env = gym.make("MuJoCoLookAt-v1")
    yield env
    env.close()


class TestLookAtEnv:
    def test_make(self, look_at_env):
        assert look_at_env is not None

    def test_obs_shape(self, look_at_env):
        obs, _ = look_at_env.reset()
        assert obs.shape == (6,)

    def test_step_five_tuple(self, look_at_env):
        look_at_env.reset()
        assert len(look_at_env.step(look_at_env.action_space.sample())) == 5

    def test_reward_range(self, look_at_env):
        look_at_env.reset()
        _, r, _, _, _ = look_at_env.step(look_at_env.action_space.sample())
        assert -0.1 <= r <= 1.0

    def test_info_keys(self, look_at_env):
        look_at_env.reset()
        _, _, _, _, info = look_at_env.step(look_at_env.action_space.sample())
        assert "orientation_error" in info
        assert "success" in info

    def test_task_description(self, look_at_env):
        assert isinstance(look_at_env.unwrapped.task_description, str)

    def test_tcp_forward_matches_gripper_direction(self, look_at_env):
        """Regression: TCP forward must equal the physical gripper direction.

        The gripperframe site has a 180° Y rotation (quat ``0 0 1 0``) so its
        local Z-axis points from the wrist toward the fingertips (``-Z`` of the
        parent body).  A previous bug used a 90° Y rotation, causing the
        forward vector to be perpendicular to the actual gripper direction and
        the look-at reward to be maximised when the arm pointed away from the
        target.
        """
        inner = look_at_env.unwrapped
        inner.reset()
        tcp_forward = inner._get_tcp_forward()
        # The parent body's -Z axis is the physical gripper direction (the
        # site offset is almost entirely along -Z of the parent body).
        body_id = inner.model.site_bodyid[inner._tcp_site_id]
        body_z = inner.data.xmat[body_id].reshape(3, 3)[:, 2]
        np.testing.assert_allclose(
            tcp_forward,
            -body_z,
            atol=1e-6,
            err_msg="TCP forward should equal -Z of parent body (gripper direction).",
        )

    def test_custom_observations(self):
        from so101_nexus_core.config import LookAtConfig
        from so101_nexus_core.observations import EndEffectorPose, JointPositions

        config = LookAtConfig(observations=[JointPositions(), EndEffectorPose()])
        env = gym.make("MuJoCoLookAt-v1", config=config)
        obs, _ = env.reset()
        assert obs.shape == (13,)  # 6 + 7
        env.close()
