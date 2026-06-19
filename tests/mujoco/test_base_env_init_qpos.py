"""Test that base_env.reset() respects options['init_qpos']."""

from __future__ import annotations

import importlib
import logging

import numpy as np
import pytest


def test_reset_uses_init_qpos_from_options():
    """Env reset should set joints to the provided init_qpos, not REST_QPOS."""
    import gymnasium as gym

    importlib.import_module("so101_nexus.mujoco")
    from so101_nexus.config import PickConfig

    env = gym.make(
        "MuJoCoPickLift-v1",
        config=PickConfig(reset_settle_frames=0),
        render_mode="rgb_array",
    )
    try:
        custom_qpos = np.array([0.1, -0.5, 0.8, 0.2, 0.0, 0.05], dtype=np.float64)
        obs, _ = env.reset(options={"init_qpos": custom_qpos})
        actual_qpos = env.unwrapped._get_current_qpos()
        np.testing.assert_allclose(actual_qpos, custom_qpos, atol=1e-6)
    finally:
        env.close()


def test_reset_with_out_of_range_init_qpos_clamps_and_warns_once(caplog):
    """Out-of-range init_qpos is clipped to actuator control bounds."""
    import gymnasium as gym

    importlib.import_module("so101_nexus.mujoco")
    from so101_nexus.config import PickConfig

    env = gym.make(
        "MuJoCoPickLift-v1",
        config=PickConfig(reset_settle_frames=0),
        render_mode="rgb_array",
    )
    try:
        inner = env.unwrapped
        huge = np.full(6, 1e9, dtype=np.float64)

        with caplog.at_level(logging.WARNING, logger="so101_nexus.mujoco.base_env"):
            env.reset(options={"init_qpos": huge})
            env.reset(options={"init_qpos": huge})

        # Clamped to the intersection of actuator ctrlrange and joint range.
        np.testing.assert_allclose(inner._get_current_qpos(), inner._target_high, atol=1e-6)
        warning_count = sum("init_qpos" in rec.message for rec in caplog.records)
        assert warning_count == 1
    finally:
        env.close()


def test_reset_without_init_qpos_uses_rest_pose():
    """Without init_qpos, reset should use the default REST_QPOS (within noise)."""
    import gymnasium as gym

    importlib.import_module("so101_nexus.mujoco")
    from so101_nexus.config import EnvironmentConfig, PickConfig

    rest = np.array(EnvironmentConfig().robot.rest_qpos_rad, dtype=np.float64)
    env = gym.make(
        "MuJoCoPickLift-v1",
        config=PickConfig(reset_settle_frames=0),
        render_mode="rgb_array",
    )
    try:
        obs, _ = env.reset(seed=0)
        inner = env.unwrapped
        actual_qpos = inner._get_current_qpos()
        # The default rest gripper (-1.1 rad) is below the menagerie gripper joint
        # limit and is clamped; compare against the model-valid rest target.
        expected = np.clip(rest, inner._target_low, inner._target_high)
        np.testing.assert_allclose(actual_qpos, expected, atol=0.025)
    finally:
        env.close()


def test_reset_with_init_pose_rest():
    """When init_pose='rest', joints should be within REST_POSE ranges."""
    import gymnasium as gym

    importlib.import_module("so101_nexus.mujoco")
    from so101_nexus.config import PickConfig, RobotConfig

    robot = RobotConfig(init_pose="rest")
    config = PickConfig(robot=robot, robot_init_qpos_noise=0.0, reset_settle_frames=0)
    env = gym.make("MuJoCoPickLift-v1", config=config, render_mode="rgb_array")
    try:
        env.reset(seed=42)
        qpos_deg = np.degrees(env.unwrapped._get_current_qpos())
        # shoulder_lift is fixed at -90
        assert qpos_deg[1] == pytest.approx(-90.0, abs=0.1)
        # Free joint: shoulder_pan within full range
        assert -110.0 <= qpos_deg[0] <= 110.0
    finally:
        env.close()


def test_reset_with_init_pose_extended():
    """When init_pose='extended', fixed joints should match EXTENDED_POSE."""
    import gymnasium as gym

    importlib.import_module("so101_nexus.mujoco")
    from so101_nexus.config import PickConfig, RobotConfig

    robot = RobotConfig(init_pose="extended")
    config = PickConfig(robot=robot, robot_init_qpos_noise=0.0, reset_settle_frames=0)
    env = gym.make("MuJoCoPickLift-v1", config=config, render_mode="rgb_array")
    try:
        env.reset(seed=42)
        qpos_deg = np.degrees(env.unwrapped._get_current_qpos())
        # shoulder_lift fixed at -30 for extended
        assert qpos_deg[1] == pytest.approx(-30.0, abs=0.1)
        # elbow_flex fixed at 20 for extended
        assert qpos_deg[2] == pytest.approx(20.0, abs=0.1)
    finally:
        env.close()


def test_reset_init_pose_none_backward_compat():
    """When init_pose=None (default), behavior is the same as before."""
    import gymnasium as gym

    importlib.import_module("so101_nexus.mujoco")
    from so101_nexus.config import EnvironmentConfig, PickConfig

    rest = np.array(EnvironmentConfig().robot.rest_qpos_rad, dtype=np.float64)
    env = gym.make(
        "MuJoCoPickLift-v1",
        config=PickConfig(reset_settle_frames=0),
        render_mode="rgb_array",
    )
    try:
        env.reset(seed=0)
        inner = env.unwrapped
        actual_qpos = inner._get_current_qpos()
        # Out-of-range default gripper rest is clamped to the model joint range.
        expected = np.clip(rest, inner._target_low, inner._target_high)
        np.testing.assert_allclose(actual_qpos, expected, atol=0.025)
    finally:
        env.close()


def _assert_within_jnt_range(env) -> None:
    m = env.model
    for jid in env._joint_ids:
        adr = m.jnt_qposadr[jid]
        lo, hi = m.jnt_range[jid]
        val = float(env.data.qpos[adr])
        assert lo - 1e-9 <= val <= hi + 1e-9, f"joint {jid} qpos {val} outside [{lo}, {hi}]"


def test_init_qpos_above_wrist_roll_limit_is_clamped_to_joint_range():
    """init_qpos with wrist_roll above the compiled joint limit is clamped.

    The menagerie wrist_roll actuator ctrlrange (~2.84121) is wider than its
    joint limit (~2.743847); a near-ctrlrange init must not land past jnt_range.
    """
    import gymnasium as gym

    importlib.import_module("so101_nexus.mujoco")
    from so101_nexus.config import PickConfig

    env = gym.make(
        "MuJoCoPickLift-v1",
        config=PickConfig(reset_settle_frames=0),
        render_mode="rgb_array",
    ).unwrapped
    try:
        qpos = np.array([0.0, 0.0, 0.0, 0.0, 2.84, 0.0], dtype=np.float64)
        env.reset(seed=0, options={"init_qpos": qpos})
        _assert_within_jnt_range(env)
    finally:
        env.close()


def test_default_rest_qpos_with_noise_stays_within_joint_range():
    """Default rest-qpos path plus reset noise must not cross the joint range.

    Regression for the post-noise clamp: a near-upper wrist_roll rest pose with
    nonzero init noise previously wrote qpos past the compiled joint limit.
    """
    import gymnasium as gym

    importlib.import_module("so101_nexus.mujoco")
    from so101_nexus.config import PickConfig, RobotConfig

    robot = RobotConfig(rest_qpos_deg=(0.0, -90.0, 90.0, 0.0, 157.0, -63.0))
    env = gym.make(
        "MuJoCoPickLift-v1",
        config=PickConfig(robot=robot, reset_settle_frames=0),
        robot_init_qpos_noise=0.1,
        render_mode="rgb_array",
    ).unwrapped
    try:
        for seed in range(20):
            env.reset(seed=seed)
            _assert_within_jnt_range(env)
    finally:
        env.close()


def test_target_delta_zero_action_holds_within_joint_range():
    """A zero action in target-delta mode must hold the reset pose, not jump.

    Regression: reset clamped wrist_roll qpos to the joint limit but previously
    seeded _prev_target with the unclamped value, so a zero action drove ctrl
    back to the (out-of-joint-range) actuator ctrlrange edge.
    """
    import gymnasium as gym

    importlib.import_module("so101_nexus.mujoco")
    from so101_nexus.config import PickConfig

    env = gym.make(
        "MuJoCoPickLift-v1",
        config=PickConfig(reset_settle_frames=0),
        control_mode="pd_joint_target_delta_pos",
        robot_init_qpos_noise=0.0,
        render_mode="rgb_array",
    ).unwrapped
    try:
        wr_jnt = env._joint_ids[4]  # wrist_roll
        wr_act = env._actuator_ids[4]
        hi = float(env.model.jnt_range[wr_jnt][1])

        # init_qpos above the joint limit but within the actuator ctrlrange.
        env.reset(seed=0, options={"init_qpos": np.array([0, 0, 0, 0, 2.84, 0.0])})
        assert env._prev_target[4] <= hi + 1e-9  # seeded with the clamped target

        env.step(np.zeros(6, dtype=np.float32))
        # Zero action must not command the actuator past the joint limit.
        assert float(env.data.ctrl[wr_act]) <= hi + 1e-9
    finally:
        env.close()
