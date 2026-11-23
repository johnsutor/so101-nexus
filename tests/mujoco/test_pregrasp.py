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


def test_approach_tilt_changes_tool_direction_at_fixed_target(env_factory):
    env = env_factory(task="PickLift", config=PickConfig(pregrasp_reset_probability=1)).unwrapped
    env.reset(seed=7)
    solver = env._pregrasp_solver
    scene = env.data.qpos.copy()
    axes = []
    positions = []
    for pitch in (0.0, 1.0):
        draw = np.array([[0.5, 0.5, 0.5, 0.5, pitch]])
        joints = solver.solve(scene, env._slots[env._target_slot_idx], draw)
        assert joints is not None
        axes.append(solver.data.site_xmat[solver.site].reshape(3, 3)[:, 2].copy())
        positions.append(solver.data.site_xpos[solver.site].copy())
    assert np.dot(*axes) < np.cos(np.deg2rad(8))
    assert np.linalg.norm(positions[0] - positions[1]) < 0.006


def test_distractor_at_candidate_arm_or_jaw_blocks_pregrasp(env_factory):
    from so101_nexus import CubeObject

    config = PickConfig(
        objects=[CubeObject(), CubeObject()], n_distractors=1, pregrasp_reset_probability=1
    )
    env = env_factory(task="PickLift", config=config).unwrapped
    env.reset(seed=7, options={"target_index": 0})
    solver = env._pregrasp_solver
    scene = env.data.qpos.copy()
    target, distractor = env._slots
    draw = np.full((1, 5), 0.5)
    assert solver.solve(scene, target, draw) is not None
    jaw = env.model.geom("fixed_jaw_box4").id
    lower_arm = env.model.body("lower_arm").id
    forearm = np.flatnonzero((env.model.geom_bodyid == lower_arm) & (env.model.geom_contype != 0))[
        0
    ]
    for obstacle_pos in solver.data.geom_xpos[[jaw, forearm]]:
        scene[distractor.qpos_addr : distractor.qpos_addr + 3] = obstacle_pos
        assert solver.solve(scene, target, draw) is None
