"""Per-task contract and behavior tests for the new batched Warp envs."""

import pytest

pytestmark = pytest.mark.warp

_ENVS = ["WarpLookAt-v1", "WarpMove-v1", "WarpPickLift-v1", "WarpPickAndPlace-v1"]
_DEFAULT_OBS_DIM = {
    "WarpLookAt-v1": 6,
    "WarpMove-v1": 6,
    "WarpPickLift-v1": 18,
    "WarpPickAndPlace-v1": 24,
}


def _make(env_id, num_envs=4, seed=0):
    import gymnasium as gym

    import so101_nexus.warp  # noqa: F401

    return gym.make_vec(
        env_id,
        num_envs=num_envs,
        device="cpu",
        seed=seed,
        vectorization_mode="vector_entry_point",
    )


@pytest.mark.parametrize("env_id", _ENVS)
def test_construction_and_obs_shape(env_id):
    import torch

    envs = _make(env_id, num_envs=4)
    obs, _ = envs.reset(seed=0)
    assert obs.shape == (4, _DEFAULT_OBS_DIM[env_id])
    assert obs.device.type == "cpu"
    assert torch.isfinite(obs).all()
    envs.close()


@pytest.mark.parametrize("env_id", _ENVS)
def test_step_shapes_and_finite_reward(env_id):
    import torch

    envs = _make(env_id, num_envs=4)
    envs.reset(seed=0)
    obs, reward, terminated, truncated, _ = envs.step(torch.zeros(envs.action_space.shape))
    assert obs.shape == (4, _DEFAULT_OBS_DIM[env_id])
    assert reward.shape == (4,)
    assert reward.dtype == torch.float32
    assert torch.isfinite(reward).all()
    assert terminated.shape == (4,)
    assert terminated.dtype == torch.bool
    assert truncated.shape == (4,)
    envs.close()


@pytest.mark.parametrize("env_id", _ENVS)
def test_seeded_reset_is_deterministic(env_id):
    import torch

    a, _ = _make(env_id, seed=7).reset(seed=7)
    b, _ = _make(env_id, seed=7).reset(seed=7)
    assert torch.allclose(a, b)


@pytest.mark.parametrize("env_id", _ENVS)
def test_step_trajectory_is_deterministic(env_id):
    import torch

    a = _make(env_id, seed=1)
    b = _make(env_id, seed=1)
    a.reset(seed=1)
    b.reset(seed=1)
    action = torch.zeros(a.action_space.shape)
    for _ in range(5):
        oa, ra, ta, _, _ = a.step(action)
        ob, rb, tb, _, _ = b.step(action)
        assert torch.allclose(oa, ob)
        assert torch.allclose(ra, rb)
        assert torch.equal(ta, tb)


def test_truncation_autoresets_world():
    import torch

    from so101_nexus.config import MoveConfig
    from so101_nexus.warp.move_env import WarpMoveVectorEnv

    env = WarpMoveVectorEnv(
        num_envs=4, config=MoveConfig(), device="cpu", max_episode_steps=2, seed=0
    )
    env.reset(seed=0)
    _, _, _, truncated, _ = env.step(torch.zeros((4, 6)))
    assert not truncated.any()
    _, _, _, truncated, _ = env.step(torch.zeros((4, 6)))
    assert truncated.all()
    assert (env._elapsed == 0).all()


def test_reset_init_qpos_applied_and_clamped():
    import torch

    from so101_nexus.config import MoveConfig
    from so101_nexus.warp.move_env import WarpMoveVectorEnv

    env = WarpMoveVectorEnv(
        num_envs=3, config=MoveConfig(reset_settle_frames=0), device="cpu", seed=0
    )
    env.reset(seed=0, options={"init_qpos": [0.0] * 6})
    q = env._joint_qpos()
    expected = torch.clamp(torch.zeros(3, 6), env._target_low, env._target_high)
    assert torch.allclose(q, expected, atol=1e-5)


def test_reset_init_pose_is_honored():
    import numpy as np
    import torch

    from so101_nexus.config import MoveConfig, Pose, RobotConfig
    from so101_nexus.warp.move_env import WarpMoveVectorEnv

    pose = Pose(
        name="p",
        shoulder_pan_deg=15.0,
        shoulder_lift_deg=-80.0,
        elbow_flex_deg=80.0,
        wrist_flex_deg=30.0,
        wrist_roll_deg=0.0,
        gripper_deg=-30.0,
    )
    env = WarpMoveVectorEnv(
        num_envs=3,
        config=MoveConfig(reset_settle_frames=0, robot=RobotConfig(init_pose=pose)),
        device="cpu",
        seed=0,
    )
    env.reset(seed=0)
    expected = torch.as_tensor(
        np.radians([15.0, -80.0, 80.0, 30.0, 0.0, -30.0]), dtype=torch.float32
    )
    expected = torch.clamp(expected.expand(3, 6), env._target_low, env._target_high)
    assert torch.allclose(env._joint_qpos(), expected, atol=1e-4)


def test_reset_bad_init_qpos_shape_raises():
    from so101_nexus.config import MoveConfig
    from so101_nexus.warp.move_env import WarpMoveVectorEnv

    env = WarpMoveVectorEnv(num_envs=3, config=MoveConfig(), device="cpu", seed=0)
    with pytest.raises(ValueError, match="init_qpos"):
        env.reset(options={"init_qpos": [0.0] * 5})


def test_move_initial_distance_equals_target():
    import torch

    from so101_nexus.config import MoveConfig
    from so101_nexus.observations import JointPositions, TargetOffset
    from so101_nexus.warp.move_env import WarpMoveVectorEnv

    env = WarpMoveVectorEnv(
        num_envs=4,
        config=MoveConfig(
            observations=[JointPositions(), TargetOffset()],
            direction="up",
            target_distance=0.1,
            reset_settle_frames=0,
        ),
        device="cpu",
        seed=0,
    )
    env.reset(seed=0)
    dist = torch.linalg.norm(env._targets - env._tcp_pos(), dim=1)
    assert torch.allclose(dist, torch.full((4,), 0.1), atol=1e-4)


def test_lookat_reward_in_unit_interval_and_orientation_error():
    import torch

    from so101_nexus.config import LookAtConfig
    from so101_nexus.warp.look_at_env import WarpLookAtVectorEnv

    env = WarpLookAtVectorEnv(num_envs=6, config=LookAtConfig(), device="cpu", seed=0)
    env.reset(seed=0)
    _, reward, _, _, info = env.step(torch.zeros((6, 6)))
    assert (reward >= 0.0).all()
    assert (reward <= 1.0).all()
    assert info["orientation_error"].shape == (6,)


def test_pick_object_pose_obs_tracks_cube():
    import torch

    from so101_nexus.config import PickConfig
    from so101_nexus.observations import ObjectPose
    from so101_nexus.warp.pick_env import WarpPickLiftVectorEnv

    env = WarpPickLiftVectorEnv(
        num_envs=4, config=PickConfig(observations=[ObjectPose()]), device="cpu", seed=0
    )
    obs, _ = env.reset(seed=0)
    assert obs.shape == (4, 7)
    assert torch.allclose(obs[:, :3], env._cube_pos(), atol=1e-5)


def test_pick_contact_budget_headroom_and_grasp_range():
    import torch

    from so101_nexus.config import PickConfig
    from so101_nexus.warp.pick_env import WarpPickLiftVectorEnv

    env = WarpPickLiftVectorEnv(num_envs=6, config=PickConfig(), device="cpu", seed=0)
    env.reset(seed=0)
    max_nacon = 0
    info = {}
    for _ in range(30):
        action = torch.clamp(torch.rand((6, 6)), env._target_low, env._target_high)
        action[:, 1] = env._target_high[1]  # drive the arm down toward the table
        _, reward, _, _, info = env.step(action)
        max_nacon = max(max_nacon, int(env._nacon_view[0]))
        assert torch.isfinite(reward).all()
    assert max_nacon < env.data.naconmax
    grasp = info["is_grasped"]
    assert ((grasp == 0.0) | (grasp == 1.0)).all()


def test_pick_rejects_unsupported_configs():
    from so101_nexus.config import PickConfig
    from so101_nexus.objects import CubeObject, YCBObject
    from so101_nexus.warp.pick_env import WarpPickLiftVectorEnv

    with pytest.raises(NotImplementedError):
        WarpPickLiftVectorEnv(
            num_envs=2, config=PickConfig(objects=YCBObject("011_banana")), device="cpu"
        )
    with pytest.raises(NotImplementedError):
        WarpPickLiftVectorEnv(
            num_envs=2,
            config=PickConfig(objects=[CubeObject(), CubeObject(color="blue")], n_distractors=1),
            device="cpu",
        )


def test_pnp_target_varies_per_world_and_respects_separation():
    import torch

    from so101_nexus.config import PickAndPlaceConfig
    from so101_nexus.warp.pick_and_place import WarpPickAndPlaceVectorEnv

    env = WarpPickAndPlaceVectorEnv(num_envs=8, config=PickAndPlaceConfig(), device="cpu", seed=0)
    env.reset(seed=0)
    target = env._target_pos()
    cube = env._cube_pos()
    assert (target[:, :2].std(dim=0) > 1e-6).any()
    sep = torch.linalg.norm(cube[:, :2] - target[:, :2], dim=1)
    assert (sep >= env.config.min_cube_target_separation - 1e-6).all()
    _, _, _, _, info = env.step(torch.zeros((8, 6)))
    assert not info["success"].any()


def test_primitive_supports_central_end_effector_pose():
    import torch

    from so101_nexus.config import MoveConfig
    from so101_nexus.observations import EndEffectorPose, JointPositions
    from so101_nexus.warp.move_env import WarpMoveVectorEnv

    env = WarpMoveVectorEnv(
        num_envs=3,
        config=MoveConfig(observations=[JointPositions(), EndEffectorPose()]),
        device="cpu",
        seed=0,
    )
    obs, _ = env.reset(seed=0)
    assert obs.shape == (3, 13)  # JointPositions(6) + EndEffectorPose(7)
    assert torch.allclose(obs[:, 6:13], env._get_tcp_pose7(), atol=1e-5)


def test_primitive_grasp_state_is_zero_without_object():
    import torch

    from so101_nexus.config import LookAtConfig
    from so101_nexus.observations import GraspState, JointPositions
    from so101_nexus.warp.look_at_env import WarpLookAtVectorEnv

    env = WarpLookAtVectorEnv(
        num_envs=4,
        config=LookAtConfig(observations=[JointPositions(), GraspState()]),
        device="cpu",
        seed=0,
    )
    obs, _ = env.reset(seed=0)
    assert obs.shape == (4, 7)  # JointPositions(6) + GraspState(1)
    assert torch.equal(obs[:, 6], torch.zeros(4))


def test_manipulation_central_obs_routed_by_base():
    import torch

    from so101_nexus.config import PickConfig
    from so101_nexus.observations import EndEffectorPose, GraspState
    from so101_nexus.warp.pick_env import WarpPickLiftVectorEnv

    env = WarpPickLiftVectorEnv(
        num_envs=2,
        config=PickConfig(observations=[EndEffectorPose(), GraspState()]),
        device="cpu",
        seed=0,
    )
    obs, _ = env.reset(seed=0)
    assert obs.shape == (2, 8)  # EndEffectorPose(7) + GraspState(1)
    assert torch.allclose(obs[:, :7], env._get_tcp_pose7(), atol=1e-5)
