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
    """The pin holds across same-step autoresets, so whole rollouts stay on target."""
    import torch

    envs = _pool_env(num_envs=2)
    try:
        envs.reset(seed=3, options={"target_index": 1})
        for _ in range(4):
            envs._task_reset(torch.ones(envs.num_envs, dtype=torch.bool))
            assert envs._target_slot.tolist() == [1, 1]
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
        force = component_slice(envs, GripperContactForce)
        assert obs.shape == (3, 15)
        drive = torch.ones(envs.action_space.shape)
        for _ in range(5):
            obs, *_ = envs.step(drive)
        torch.testing.assert_close(
            obs[:, effort],
            envs._qfrc_actuator.index_select(1, envs._dof_adr).to(torch.float32),
        )
        torch.testing.assert_close(obs[:, force], envs._gripper_contact_force().to(torch.float32))
        assert torch.isfinite(obs).all()
    finally:
        envs.close()
