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


def test_gripper_contact_force_is_nonzero_while_the_jaw_squeezes():
    """Cross-checked against mujoco_warp's own world-frame contact force.

    Comparing the observation against ``_gripper_contact_force()`` would compare
    the reader with itself and pass for a transposed frame or a flipped sign;
    ``mjw.contact_force(..., to_world_frame=True)`` is an independent reference.
    """
    import mujoco_warp as mjw
    import torch
    import warp as wp

    from so101_nexus.observations import GripperContactForce, JointPositions
    from so101_nexus.testing import component_slice

    envs = _pool_env(num_envs=2, observations=[JointPositions(), GripperContactForce()])
    try:
        envs.reset(seed=0)
        force_slice = component_slice(envs, GripperContactForce)
        close = torch.zeros(envs.action_space.shape)
        close[:, -1] = envs._target_low[-1]
        obs = None
        # Hold the target between the fingers while the jaw closes on it.
        for _ in range(25):
            cols = envs._target_qadr[:, None] + torch.arange(3, device=envs.device)
            envs.qpos[envs._world_rows[:, None], cols] = envs._tcp_pos()
            obs, *_ = envs.step(close)
        assert obs is not None
        observed = obs[:, force_slice]
        assert observed.abs().max() > 0.0

        # Independent reference: world-frame contact forces summed with the same
        # exactly-one-finger sign rule.
        envs._ensure_contact_force_buffers()
        with wp.ScopedDevice(envs._wp_device):
            mjw.contact_force(envs.model, envs.data, envs._contact_ids, True, envs._force_buf)
        world_force = wp.to_torch(envs._force_buf)[:, :3]
        nacon = int(envs._nacon_view[0])
        geom = envs._contact_geom_view[:nacon].long()
        worldid = envs._contact_world_view[:nacon].long()
        finger = envs._gripper_mask | envs._jaw_mask
        sign = finger[geom[:, 1]].float() - finger[geom[:, 0]].float()
        expected = torch.zeros_like(observed)
        expected.scatter_add_(
            0,
            worldid.unsqueeze(1).expand(-1, 3),
            world_force[:nacon] * sign.unsqueeze(1),
        )
        torch.testing.assert_close(observed, expected, atol=1e-4, rtol=1e-4)
    finally:
        envs.close()


@pytest.mark.parametrize("threshold,expected", [(0.3, 0.0), (-1.0, 1.0)])
def test_grasp_opposing_normal_threshold_changes_the_warp_verdict(threshold, expected):
    """The straddle rejection must be driven by the config field on Warp too.

    A cube far too wide for the jaw is held at the TCP while the gripper closes,
    so both finger sets press the same side of it. The default threshold must
    reject that; ``-1.0``, the documented escape hatch back to bilateral contact
    alone, must accept it. Same physics, only the config differs.
    """
    import torch

    from so101_nexus.config import PickConfig, RobotConfig
    from so101_nexus.objects import CubeObject
    from so101_nexus.warp.pick_env import WarpPickLiftVectorEnv

    config = PickConfig(
        objects=[CubeObject(half_size=0.05, color="red")],
        n_distractors=0,
        robot=RobotConfig(grasp_opposing_normal_threshold=threshold),
    )
    envs = WarpPickLiftVectorEnv(num_envs=2, config=config, device="cpu", seed=0)
    try:
        envs.reset(seed=0)
        close = torch.zeros(envs.action_space.shape)
        close[:, -1] = envs._target_low[-1]
        for _ in range(25):
            cols = envs._target_qadr[:, None] + torch.arange(3, device=envs.device)
            envs.qpos[envs._world_rows[:, None], cols] = envs._tcp_pos()
            envs.step(close)
        assert envs._is_grasping().tolist() == [expected, expected]
    finally:
        envs.close()
