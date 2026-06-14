"""Tests for the menagerie-backed ManiSkill SO101 agent."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch

import so101_nexus_maniskill  # noqa: F401 - registers envs
from so101_nexus_core.config import ReachConfig, RobotConfig
from so101_nexus_maniskill import menagerie_constants as mc


def test_agent_loads_menagerie_link_names(so101_reach_env):
    robot = so101_reach_env.agent.robot
    names = {link.name for link in robot.links}
    assert {"gripper", "moving_jaw_so101_v1", "camera_mount"} <= names
    assert "gripper_link" not in names
    assert "gripper_frame_link" not in names


def test_agent_active_joint_order(so101_reach_env):
    robot = so101_reach_env.agent.robot
    assert [j.name for j in robot.active_joints] == [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    ]


def test_rest_keyframe_derives_from_core(so101_reach_env):
    expected = np.radians(RobotConfig().rest_qpos_deg)
    np.testing.assert_allclose(so101_reach_env.agent.keyframes["rest"].qpos, expected, atol=1e-9)


def test_tcp_pose_composes_offset_on_gripper_link(so101_reach_env):
    env = so101_reach_env
    env.reset(seed=0)
    gripper = env.agent.robot.links_map["gripper"]
    from mani_skill.utils.structs.pose import Pose

    offset = Pose.create_from_pq(
        p=torch.tensor(mc.TCP_OFFSET_POS, dtype=torch.float32),
        q=torch.tensor(mc.TCP_OFFSET_QUAT, dtype=torch.float32),
    )
    expected = gripper.pose * offset
    np.testing.assert_allclose(
        env.agent.tcp_pose.raw_pose.cpu().numpy(), expected.raw_pose.cpu().numpy(), atol=1e-5
    )


def _assert_menagerie_patches_applied(robot) -> None:
    """Assert inertials, gripper friction, and armature are applied on every
    per-scene PhysX object of ``robot``. Works for any num_envs (each link's
    ``_objs`` has one entry per sub-scene)."""
    from transforms3d.quaternions import quat2mat

    for name, inertial in mc.LINK_INERTIALS.items():
        link = robot.links_map[name]
        for obj in link._objs:
            assert obj.mass == pytest.approx(inertial.mass, rel=1e-6), name
            np.testing.assert_allclose(
                np.asarray(obj.cmass_local_pose.p), inertial.com_pos, atol=1e-9
            )
            rot_applied = quat2mat(np.asarray(obj.cmass_local_pose.q))
            tensor_applied = rot_applied @ np.diag(np.asarray(obj.inertia)) @ rot_applied.T
            rot_intended = quat2mat(inertial.principal_quat)
            tensor_intended = rot_intended @ np.diag(inertial.principal_moments) @ rot_intended.T
            np.testing.assert_allclose(tensor_applied, tensor_intended, atol=1e-10)

    for name in mc.GRIPPER_FRICTION_LINKS:
        link = robot.links_map[name]
        for obj in link._objs:
            shapes = obj.get_collision_shapes()
            assert shapes, f"{name} has no collision shapes"
            for shape in shapes:
                mat = shape.get_physical_material()
                assert mat.get_static_friction() == pytest.approx(mc.GRIPPER_STATIC_FRICTION)
                assert mat.get_dynamic_friction() == pytest.approx(mc.GRIPPER_DYNAMIC_FRICTION)
                assert shape.get_patch_radius() == pytest.approx(mc.GRIPPER_PATCH_RADIUS)
                assert shape.get_min_patch_radius() == pytest.approx(mc.GRIPPER_MIN_PATCH_RADIUS)

    for joint in robot.active_joints:
        for jobj in joint._objs:
            np.testing.assert_allclose(np.asarray(jobj.armature), [mc.JOINT_ARMATURE], atol=1e-9)


def test_menagerie_patches_applied_single_env(so101_reach_env):
    """Patches apply on the default single (CPU PhysX) env. This runs in CPU-only
    CI - which skips the num_envs=2 checks below - so the fidelity patches always
    have direct CI coverage; the vectorized tests add the every-sub-scene guarantee
    when a GPU is available."""
    _assert_menagerie_patches_applied(so101_reach_env.agent.robot)


def test_link_masses_applied_on_every_subscene(so101_reach_env_vec):
    robot = so101_reach_env_vec.agent.robot
    for name, inertial in mc.LINK_INERTIALS.items():
        link = robot.links_map[name]
        for obj in link._objs:
            assert obj.mass == pytest.approx(inertial.mass, rel=1e-6), name


def test_link_inertia_and_com_applied(so101_reach_env_vec):
    from transforms3d.quaternions import quat2mat

    robot = so101_reach_env_vec.agent.robot
    for name, inertial in mc.LINK_INERTIALS.items():
        link = robot.links_map[name]
        for obj in link._objs:
            np.testing.assert_allclose(
                sorted(np.asarray(obj.inertia)), sorted(inertial.principal_moments), atol=1e-10
            )
            np.testing.assert_allclose(
                np.asarray(obj.cmass_local_pose.p), inertial.com_pos, atol=1e-9
            )
            # The principal-axis orientation is load-bearing: compare the full
            # applied inertia tensor (moments rotated by the applied cmass quat)
            # to the intended tensor, so a wrong quat cannot pass on sorted
            # moments alone. (Comparing quats directly is sign/axis ambiguous.)
            applied_q = np.asarray(obj.cmass_local_pose.q)  # (w, x, y, z)
            rot_applied = quat2mat(applied_q)
            tensor_applied = rot_applied @ np.diag(np.asarray(obj.inertia)) @ rot_applied.T
            rot_intended = quat2mat(inertial.principal_quat)
            tensor_intended = rot_intended @ np.diag(inertial.principal_moments) @ rot_intended.T
            np.testing.assert_allclose(tensor_applied, tensor_intended, atol=1e-10)


def test_gripper_friction_applied_on_every_collision_shape(so101_reach_env_vec):
    robot = so101_reach_env_vec.agent.robot
    for name in mc.GRIPPER_FRICTION_LINKS:
        link = robot.links_map[name]
        for obj in link._objs:
            shapes = obj.get_collision_shapes()
            assert shapes, f"{name} has no collision shapes"
            for shape in shapes:
                mat = shape.get_physical_material()
                assert mat.get_static_friction() == pytest.approx(mc.GRIPPER_STATIC_FRICTION)
                assert mat.get_dynamic_friction() == pytest.approx(mc.GRIPPER_DYNAMIC_FRICTION)
                assert shape.get_patch_radius() == pytest.approx(mc.GRIPPER_PATCH_RADIUS)
                assert shape.get_min_patch_radius() == pytest.approx(mc.GRIPPER_MIN_PATCH_RADIUS)


def test_joint_armature_applied(so101_reach_env_vec):
    robot = so101_reach_env_vec.agent.robot
    for joint in robot.active_joints:
        for jobj in joint._objs:
            np.testing.assert_allclose(np.asarray(jobj.armature), [mc.JOINT_ARMATURE], atol=1e-9)


def test_qlimits_equal_menagerie_joint_limits(so101_reach_env_vec):
    robot = so101_reach_env_vec.agent.robot
    qlimits = robot.get_qlimits()[0].cpu().numpy()  # (dof, 2)
    for i, joint in enumerate(robot.active_joints):
        lo, hi = mc.JOINT_LIMITS[joint.name]
        np.testing.assert_allclose(qlimits[i], [lo, hi], atol=1e-5)


def test_pd_joint_pos_action_space_equals_joint_limits():
    """The absolute (non-delta) controller's action space must equal the
    menagerie joint limits (lower/upper=None resolves to compiled limits)."""
    env = gym.make(
        "ManiSkillReachSO101-v1",
        config=ReachConfig(),
        num_envs=1,
        obs_mode="state",
        render_mode=None,
        control_mode="pd_joint_pos",
    )
    try:
        space = env.unwrapped.single_action_space
        joints = [j.name for j in env.unwrapped.agent.robot.active_joints]
        lows = np.array([mc.JOINT_LIMITS[n][0] for n in joints], dtype=np.float32)
        highs = np.array([mc.JOINT_LIMITS[n][1] for n in joints], dtype=np.float32)
        np.testing.assert_allclose(space.low, lows, atol=1e-4)
        np.testing.assert_allclose(space.high, highs, atol=1e-4)
    finally:
        env.close()


def test_delta_action_space_is_normalized():
    """Delta modes keep ManiSkill's default normalized [-1, 1] action space
    (unchanged from the pre-menagerie behavior). The +/-0.05 arm / +/-0.2
    gripper delta scale is the controller's internal range, not the action
    space; normalize_action stays at its default (True) for delta modes."""
    for mode in ("pd_joint_delta_pos", "pd_joint_target_delta_pos"):
        env = gym.make(
            "ManiSkillReachSO101-v1",
            config=ReachConfig(),
            num_envs=1,
            obs_mode="state",
            render_mode=None,
            control_mode=mode,
        )
        try:
            space = env.unwrapped.single_action_space
            np.testing.assert_allclose(space.low, [-1.0] * 6, atol=1e-6)
            np.testing.assert_allclose(space.high, [1.0] * 6, atol=1e-6)
        finally:
            env.close()


_CONTROL_MODES = ["pd_joint_pos", "pd_joint_delta_pos", "pd_joint_target_delta_pos"]


def _assert_drive_state(robot):
    for joint in robot.active_joints:
        for jobj in joint._objs:
            assert jobj.stiffness == pytest.approx(mc.DRIVE_STIFFNESS, rel=1e-4)
            assert jobj.damping == pytest.approx(mc.DRIVE_DAMPING, rel=1e-4)
            assert jobj.force_limit == pytest.approx(mc.FORCE_LIMIT, rel=1e-4)
            assert jobj.friction == pytest.approx(mc.JOINT_FRICTIONLOSS, rel=1e-4)
            np.testing.assert_allclose(np.asarray(jobj.armature), [mc.JOINT_ARMATURE], atol=1e-6)


@pytest.mark.parametrize("control_mode", _CONTROL_MODES)
def test_controller_drive_state(control_mode):
    env = gym.make(
        "ManiSkillReachSO101-v1",
        config=ReachConfig(),
        num_envs=1,
        obs_mode="state",
        render_mode=None,
        control_mode=control_mode,
    )
    try:
        env.reset()
        _assert_drive_state(env.unwrapped.agent.robot)
    finally:
        env.close()


def test_controller_drive_state_survives_mode_switches():
    env = gym.make(
        "ManiSkillReachSO101-v1",
        config=ReachConfig(),
        num_envs=1,
        obs_mode="state",
        render_mode=None,
        control_mode="pd_joint_pos",
    )
    try:
        env.reset()
        robot = env.unwrapped.agent.robot
        for mode in _CONTROL_MODES:
            env.unwrapped.agent.set_control_mode(mode)
            env.reset()
            _assert_drive_state(robot)
    finally:
        env.close()


# --- Wrist camera cross-backend parity --------------------------------------
# Pinned from the MuJoCo backend wrist camera at zero qpos, pitch 37.5deg, zero
# noise, after the cam_pos/cam_quat fix so the camera tracks the WristCamera
# component. World optical axes in MuJoCo convention: forward = -cam_xmat[:, 2],
# up = cam_xmat[:, 1]. SAPIEN convention: forward = +x_local, up = +z_local
# (confirmed via sapien_utils.look_at). Comparing world axes (not raw quats)
# avoids the cross-convention mismatch so a mirrored/backward image cannot pass.
_MJ_WRIST_CAM_POS = (0.33330457, 0.03977481, 0.23604647)
_MJ_WRIST_CAM_FWD = (0.793387, 0.60804, 0.028695)
_MJ_WRIST_CAM_UP = (-0.608716, 0.792414, 0.039316)


def _wrist_cam_world_axes(env):
    """Return (pos, forward, up) world axes of the built SO101 wrist camera."""
    from transforms3d.quaternions import quat2mat

    gp = env.unwrapped._sensors["wrist_camera"].camera.global_pose
    pos = np.asarray(gp.p).reshape(-1)[:3]
    rot = quat2mat(np.asarray(gp.q).reshape(-1)[:4])
    return pos, rot[:, 0], rot[:, 2]  # SAPIEN forward=+x_local, up=+z_local


def test_wrist_camera_world_pose_matches_mujoco_backend():
    from so101_nexus_core.observations import JointPositions, WristCamera

    cfg = ReachConfig(
        observations=[
            JointPositions(),
            WristCamera(
                width=64,
                height=64,
                pos_x_noise=0.0,
                pos_y_noise=0.0,
                pos_z_noise=0.0,
                pitch_deg_range=(37.5, 37.5),
                fov_deg_range=(60.0, 60.0),
            ),
        ]
    )
    env = gym.make(
        "ManiSkillReachSO101-v1", config=cfg, num_envs=1, obs_mode="rgbd", render_mode=None
    )
    try:
        env.reset(seed=0, options={"init_qpos": np.zeros(6)})
        pos, fwd, up = _wrist_cam_world_axes(env)
        np.testing.assert_allclose(pos, _MJ_WRIST_CAM_POS, atol=2e-3)
        assert float(np.dot(fwd, _MJ_WRIST_CAM_FWD)) == pytest.approx(1.0, abs=2e-3)
        assert float(np.dot(up, _MJ_WRIST_CAM_UP)) == pytest.approx(1.0, abs=2e-3)
    finally:
        env.close()


def _wrist_cam_world_pos(pos_y_center: float, pos_z_center: float) -> np.ndarray:
    """SO101 wrist camera world position at zero qpos for given pos centers."""
    from so101_nexus_core.observations import JointPositions, WristCamera

    cfg = ReachConfig(
        observations=[
            JointPositions(),
            WristCamera(
                width=64,
                height=64,
                pos_x_noise=0.0,
                pos_y_noise=0.0,
                pos_z_noise=0.0,
                pos_y_center=pos_y_center,
                pos_z_center=pos_z_center,
                pitch_deg_range=(0.0, 0.0),
                fov_deg_range=(60.0, 60.0),
            ),
        ]
    )
    env = gym.make(
        "ManiSkillReachSO101-v1", config=cfg, num_envs=1, obs_mode="rgbd", render_mode=None
    )
    try:
        env.reset(seed=0, options={"init_qpos": np.zeros(6)})
        return _wrist_cam_world_axes(env)[0]
    finally:
        env.close()


def test_wrist_camera_consumes_component_pos_centers():
    """Non-default WristCamera pos centers must move the SO101 wrist camera.

    Proves the SO101 path reads the WristCamera component (not the static
    preset). The mount orientation is fixed at a given qpos, so a rigid change
    of the local center shifts the world position by the same Euclidean
    distance. The old preset-only path would not move at all.
    """
    base = _wrist_cam_world_pos(pos_y_center=0.04, pos_z_center=-0.04)
    shifted = _wrist_cam_world_pos(pos_y_center=0.10, pos_z_center=-0.10)
    expected_delta = float(np.linalg.norm([0.0, 0.06, -0.06]))
    assert float(np.linalg.norm(shifted - base)) == pytest.approx(expected_delta, abs=1e-3)


def test_wrist_camera_randomization_is_seeded():
    """env.reset(seed=...) must reproduce the wrist-camera pose.

    The pose is sampled with the seeded episode RNG (matching the MuJoCo
    backend's self.np_random), so the same seed yields the same camera and
    different seeds differ. Regression for global-np.random sampling that
    ignored the reset seed and broke cross-backend determinism.
    """
    from so101_nexus_core.observations import JointPositions, WristCamera

    def cam_pose(seed: int) -> np.ndarray:
        cfg = ReachConfig(observations=[JointPositions(), WristCamera(width=64, height=64)])
        env = gym.make(
            "ManiSkillReachSO101-v1", config=cfg, num_envs=1, obs_mode="rgbd", render_mode=None
        )
        try:
            env.reset(seed=seed)
            gp = env.unwrapped._sensors["wrist_camera"].camera.global_pose
            return np.concatenate(
                [np.asarray(gp.p).reshape(-1)[:3], np.asarray(gp.q).reshape(-1)[:4]]
            )
        finally:
            env.close()

    same_a, same_b, other = cam_pose(123), cam_pose(123), cam_pose(999)
    np.testing.assert_allclose(same_a, same_b, atol=1e-6)
    assert not np.allclose(same_a, other, atol=1e-4)


# --- Cross-backend TCP parity (zero-qpos and default rest) -------------------
# Pinned by the MuJoCo backend test_rest_tcp_pose_matches_menagerie at zero qpos.
_MJ_ZERO_QPOS_TCP_POS = (0.3914432501, -0.0009794699, 0.2460073072)
_MJ_ZERO_QPOS_TCP_QUAT = (0.7064841888, 0.0172191552, 0.7073102466, 0.0171990353)
# Pinned from the MuJoCo backend at the default clamped rest pose, zero noise.
_MJ_DEFAULT_REST_TCP_POS = (0.2246947757, -0.0009794652, 0.0616355396)
_MJ_DEFAULT_REST_TCP_QUAT = (0.4396896602, 0.0218570366, 0.8978199686, 0.0107040535)


def test_zero_qpos_tcp_pose_matches_mujoco_backend(so101_reach_env):
    env = so101_reach_env
    env.reset(seed=0, options={"init_qpos": np.zeros(6)})
    tcp = env.agent.tcp_pose
    pos = tcp.p[0].cpu().numpy()
    quat = tcp.q[0].cpu().numpy()  # ManiSkill pose quats are (w, x, y, z)
    np.testing.assert_allclose(pos, _MJ_ZERO_QPOS_TCP_POS, atol=1e-3)
    dot = abs(float(np.dot(quat, _MJ_ZERO_QPOS_TCP_QUAT)))
    assert dot == pytest.approx(1.0, abs=2e-3), (quat, _MJ_ZERO_QPOS_TCP_QUAT)


def test_default_rest_tcp_pose_matches_mujoco_backend():
    env = gym.make(
        "ManiSkillReachSO101-v1",
        config=ReachConfig(reset_settle_frames=0),
        num_envs=1,
        obs_mode="state",
        render_mode=None,
    )
    try:
        inner = env.unwrapped
        inner.robot_init_qpos_noise = 0.0
        inner.reset(seed=0)  # default rest path, clamped to qlimits
        tcp = inner.agent.tcp_pose
        pos = tcp.p[0].cpu().numpy()
        quat = tcp.q[0].cpu().numpy()
        np.testing.assert_allclose(pos, _MJ_DEFAULT_REST_TCP_POS, atol=2e-3)
        dot = abs(float(np.dot(quat, _MJ_DEFAULT_REST_TCP_QUAT)))
        assert dot == pytest.approx(1.0, abs=2e-3), (quat, _MJ_DEFAULT_REST_TCP_QUAT)
    finally:
        env.close()
