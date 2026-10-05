"""Warp tests for reset(options={'target_index': ...}) and the force observations."""

import pytest

pytestmark = pytest.mark.warp

_COLORS = ("red", "blue", "green")


def _pool_env(num_envs=4, n_distractors=1, observations=None):
    from so101_nexus.config import PickConfig
    from so101_nexus.objects import CubeObject
    from so101_nexus.warp.pick_env import WarpPickLiftVectorEnv

    config = PickConfig(
        objects=[CubeObject(half_size=0.02, color=c) for c in _COLORS],
        n_distractors=n_distractors,
        observations=observations,
    )
    return WarpPickLiftVectorEnv(num_envs=num_envs, config=config, device="cpu", seed=0)


def test_scalar_target_index_pins_every_world():
    envs = _pool_env()
    try:
        for k in range(len(_COLORS)):
            _, _ = envs.reset(seed=3, options={"target_index": k})
            assert envs._target_slot.tolist() == [k] * envs.num_envs
    finally:
        envs.close()


def test_per_world_target_index_is_honoured():
    import torch

    envs = _pool_env(num_envs=3)
    try:
        envs.reset(seed=3, options={"target_index": torch.tensor([2, 0, 1])})
        assert envs._target_slot.tolist() == [2, 0, 1]
    finally:
        envs.close()


def test_target_index_only_relabels_when_the_slot_is_already_active():
    """Counterfactual pairs need byte-identical scenes across the two targets."""
    import torch

    # Every pool slot is active, so a pin can never displace another object.
    envs = _pool_env(num_envs=2, n_distractors=len(_COLORS) - 1)
    try:
        envs.reset(seed=7, options={"target_index": 0})
        first = envs.qpos.clone()
        envs.reset(seed=7, options={"target_index": 2})
        assert envs._target_slot.tolist() == [2, 2]
        torch.testing.assert_close(envs.qpos, first)
    finally:
        envs.close()


def test_target_index_survives_autoreset():
    """The pin holds across same-step autoresets, so whole rollouts stay on target.

    Driven through real ``step()`` truncations rather than by calling
    ``_task_reset`` directly: the point is that the autoreset path inside
    ``step`` reaches the pin, which a direct helper call cannot show.
    """
    import torch

    envs = _pool_env(num_envs=2)
    envs.max_episode_steps = 3
    try:
        envs.reset(seed=3, options={"target_index": 1})
        zeros = torch.zeros(envs.action_space.shape)
        truncations = 0
        for _ in range(9):
            _, _, _, truncated, info = envs.step(zeros)
            truncations += int(truncated.any())
            assert info["target_index"].tolist() == [1, 1]
        assert truncations >= 2
        envs.reset(seed=3)
        assert envs._target_index_override is None
    finally:
        envs.close()


def test_target_index_is_reported_in_info():
    envs = _pool_env(num_envs=2)
    try:
        import torch

        envs.reset(seed=3, options={"target_index": 2})
        _, _, _, _, info = envs.step(torch.zeros(envs.action_space.shape))
        assert info["target_index"].tolist() == [2, 2]
    finally:
        envs.close()


@pytest.mark.parametrize("bad", [3, -1])
def test_out_of_range_target_index_raises(bad):
    envs = _pool_env()
    try:
        with pytest.raises(ValueError, match="target_index"):
            envs.reset(seed=0, options={"target_index": bad})
    finally:
        envs.close()


def test_target_index_on_a_poolless_task_raises():
    import gymnasium as gym

    import so101_nexus.warp  # noqa: F401

    envs = gym.make_vec(
        "WarpMove-v1", num_envs=2, device="cpu", seed=0, vectorization_mode="vector_entry_point"
    )
    try:
        with pytest.raises(ValueError, match="no object pool"):
            envs.reset(seed=0, options={"target_index": 0})
    finally:
        envs.close()


def test_force_observations_match_the_simulator_state():
    """JointEfforts and GripperContactForce are live reads, batched like the rest."""
    import torch

    from so101_nexus.observations import (
        GripperContactForce,
        JointEfforts,
        JointPositions,
    )
    from so101_nexus.testing import component_slice

    envs = _pool_env(
        num_envs=3,
        observations=[JointPositions(), JointEfforts(), GripperContactForce()],
    )
    try:
        obs, _ = envs.reset(seed=0)
        effort = component_slice(envs, JointEfforts)
        assert obs.shape == (3, 15)
        drive = torch.ones(envs.action_space.shape)
        for _ in range(5):
            obs, *_ = envs.step(drive)
        torch.testing.assert_close(
            obs[:, effort],
            envs._qfrc_actuator.index_select(1, envs._dof_adr).to(torch.float32),
        )
        assert obs[:, effort].abs().max() > 0.0
        assert torch.isfinite(obs).all()
    finally:
        envs.close()


def test_gripper_contact_force_matches_loaded_contacts(env_factory):
    """Compare force observations with Warp's independent world-frame reference."""
    import mujoco_warp as mjw
    import torch
    import warp as wp

    from so101_nexus import CubeObject, PickConfig
    from so101_nexus.observations import GripperContactForce, JointPositions
    from so101_nexus.testing import component_slice

    half_size = 0.05
    env = env_factory(
        "warp",
        task="PickLift",
        config=PickConfig(
            objects=[CubeObject(half_size=half_size)],
            observations=[JointPositions(), GripperContactForce()],
        ),
    ).unwrapped
    obs, _ = env.reset(seed=0)
    force_slice = component_slice(env, GripperContactForce)
    torch.testing.assert_close(obs[:, force_slice], torch.zeros((env.num_envs, 3)))

    _press_fingers_into_target(env, half_size)
    observed = env._compute_obs()[:, force_slice]
    assert (observed.abs().amax(dim=1) > 0.0).all()

    with wp.ScopedDevice(env._wp_device):
        mjw.contact_force(env.model, env.data, env._contact_ids, True, env._force_buf)
    world_force = wp.to_torch(env._force_buf)[:, :3]
    nacon = int(env._nacon_view[0])
    geom = env._contact_geom_view[:nacon].long()
    worldid = env._contact_world_view[:nacon].long()
    finger = env._gripper_mask | env._jaw_mask
    sign = finger[geom[:, 1]].float() - finger[geom[:, 0]].float()
    expected = torch.zeros_like(observed)
    expected.scatter_add_(
        0,
        worldid.unsqueeze(1).expand(-1, 3),
        world_force[:nacon] * sign.unsqueeze(1),
    )
    torch.testing.assert_close(observed, expected, atol=1e-4, rtol=1e-4)


def test_grasp_opposing_normal_threshold_changes_the_warp_verdict(env_factory):
    """Both thresholds evaluate the same force-bearing, same-face contacts."""

    from so101_nexus import CubeObject, PickConfig

    half_size = 0.05
    env = env_factory(
        "warp", task="PickLift", config=PickConfig(objects=[CubeObject(half_size=half_size)])
    ).unwrapped
    env.reset(seed=0)
    _press_fingers_into_target(env, half_size)

    assert env._is_grasping().tolist() == [0.0, 0.0]
    env.config.robot.grasp_opposing_normal_threshold = -1.0
    assert env._is_grasping().tolist() == [1.0, 1.0]


def _press_fingers_into_target(env, half_size):
    """Load stationary finger contacts without letting the target escape."""
    import mujoco_warp as mjw
    import torch
    import warp as wp

    close = env._joint_qpos().clone()
    close[:, -1] = env._target_low[-1]
    for _ in range(25):
        env.step(close)

    fingers = wp.to_torch(env.data.geom_xpos)[:, env._gripper_mask | env._jaw_mask]
    pose = torch.zeros((env.num_envs, 7), device=env.device)
    pose[:, :3] = fingers.mean(1)
    # Match the native straddle test's 40 mm penetration to load both finger sets.
    pose[:, 0] = fingers[:, :, 0].max(1).values + half_size - 0.04
    pose[:, 3] = 1.0
    cols = env._target_qadr[:, None] + torch.arange(7, device=env.device)
    env.qpos[env._world_rows[:, None], cols] = pose
    env.qvel.zero_()
    with wp.ScopedDevice(env._wp_device):
        mjw.forward(env.model, env.data)
    env._contact_cache = None
