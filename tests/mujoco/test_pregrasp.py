"""Object-relative PickLift reset coverage."""

import numpy as np

from so101_nexus import PickConfig


def test_picklift_pregrasp(env_factory):
    env = env_factory(task="PickLift", config=PickConfig(pregrasp_reset_probability=1)).unwrapped
    poses = []
    for seed in (1, 7, 1):
        _, info = env.reset(seed=seed)
        poses.append(env._get_tcp_pose()[:3] - env._get_target_pose()[:3])
        assert np.linalg.norm(poses[-1][:2]) < 0.025
        assert 0.015 < poses[-1][2] < 0.09
        assert env.data.site_xmat[env._tcp_site_id, 8] < -0.64
        assert env.data.qpos[env._qpos_addrs[-1]] > 1.0
        assert not env._pregrasp_solver.robot_geom[env.data.contact.geom].any()
        assert not info["is_grasped"]
        assert not info["success"]
        assert np.array_equal(env._prev_target, env.data.ctrl[env._actuator_ids])
    np.testing.assert_array_equal(poses[0], poses[2])
    assert not np.allclose(poses[0], poses[1])


def test_pregrasp_explicit_pose_wins(env_factory):
    config = PickConfig(pregrasp_reset_probability=1, reset_settle_frames=0)
    env = env_factory(task="PickLift", config=config).unwrapped
    env.reset(seed=8, options={"init_qpos": np.zeros(6)})
    np.testing.assert_array_equal(env.data.ctrl[env._actuator_ids], np.zeros(6))


def test_unreachable_pregrasp_preserves_normal_reset(env_factory):
    kwargs = {"spawn_center": (2, 0)}
    default = env_factory(task="PickLift", config=PickConfig(**kwargs)).unwrapped
    pregrasp = env_factory(
        task="PickLift", config=PickConfig(pregrasp_reset_probability=1, **kwargs)
    ).unwrapped
    default.reset(seed=3)
    pregrasp.reset(seed=3)
    np.testing.assert_array_equal(pregrasp.data.ctrl, default.data.ctrl)
    np.testing.assert_array_equal(pregrasp.data.qpos, default.data.qpos)
