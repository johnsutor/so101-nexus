"""Public reset and action inputs retain their meaning across backends."""

import numpy as np
import pytest

from so101_nexus import (
    CubeObject,
    JointPositions,
    LookAtConfig,
    MoveConfig,
    PickAndPlaceConfig,
    PickAndPlaceV2Config,
    PickConfig,
    PickReturnConfig,
    StackCubeConfig,
    TouchConfig,
)

_TASKS = [
    ("Touch", 1, TouchConfig),
    ("LookAt", 1, LookAtConfig),
    ("Move", 1, MoveConfig),
    ("PickLift", 1, PickConfig),
    ("PickReturn", 1, PickReturnConfig),
    ("PickAndPlace", 1, PickAndPlaceConfig),
    ("PickAndPlace", 2, PickAndPlaceV2Config),
    ("StackCube", 1, StackCubeConfig),
]


@pytest.mark.parametrize("task,version,config_cls", _TASKS)
@pytest.mark.parametrize("noise", [0.0, 0.1])
def test_mujoco_reset_uses_config_noise(env_factory, task, version, config_cls, noise):
    config = config_cls(
        observations=[JointPositions()], reset_settle_frames=0, robot_init_qpos_noise=noise
    )
    env = env_factory(task=task, version=version, config=config)
    first, _ = env.reset(seed=8)
    repeated, _ = env.reset(seed=8)
    np.testing.assert_array_equal(first, repeated)
    rest = np.clip(config.robot.rest_qpos_rad, env.action_space.low, env.action_space.high)
    if noise == 0:
        np.testing.assert_allclose(first, rest, atol=1e-7)
    else:
        assert np.max(np.abs(first - rest)) > 0.02
        np.testing.assert_array_less(np.abs(first - rest), noise + 1e-7)


@pytest.mark.parametrize("task,version,config_cls", _TASKS)
@pytest.mark.parametrize("noise", [0.0, 0.1])
def test_mujoco_explicit_noise_overrides_config(env_factory, task, version, config_cls, noise):
    config = config_cls(
        observations=[JointPositions()], reset_settle_frames=0, robot_init_qpos_noise=0.2
    )
    overridden = env_factory(task=task, version=version, config=config, robot_init_qpos_noise=noise)
    actual, _ = overridden.reset(seed=8)
    rest = np.clip(
        config.robot.rest_qpos_rad, overridden.action_space.low, overridden.action_space.high
    )
    np.testing.assert_array_less(np.abs(actual - rest), noise + 1e-7)
    if noise:
        assert np.max(np.abs(actual - rest)) > 0.02


@pytest.mark.parametrize("backend", ["mujoco", "warp"])
@pytest.mark.parametrize("task", ["PickLift", "Move"])
def test_none_target_index_matches_omitted_option(env_factory, backend, task):
    env = env_factory(backend, task=task)
    expected, _ = env.reset(seed=8)
    actual, _ = env.reset(seed=8, options={"target_index": None})
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("backend", ["mujoco", "warp"])
@pytest.mark.parametrize("target_index", [0.9, 1.0, False, True, "0", np.bool_(False)])
def test_target_index_rejects_nonintegers(env_factory, backend, target_index):
    config = PickConfig(objects=[CubeObject(color="red"), CubeObject(color="blue")])
    env = env_factory(backend, task="PickLift", config=config)
    with pytest.raises(ValueError, match="target_index"):
        env.reset(seed=8, options={"target_index": target_index})


@pytest.mark.parametrize("target_index", [[0.0, 1.0], [False, 1], np.array([True, False])])
def test_warp_target_index_rejects_noninteger_vectors(env_factory, target_index):
    config = PickConfig(objects=[CubeObject(color="red"), CubeObject(color="blue")])
    env = env_factory("warp", task="PickLift", config=config)
    with pytest.raises(ValueError, match="target_index"):
        env.reset(seed=8, options={"target_index": target_index})


@pytest.mark.parametrize("target_index", [np.int64(1), [1, 0], np.array([1, 0], dtype=np.int32)])
def test_warp_target_index_accepts_integer_scalars_and_vectors(env_factory, target_index):
    config = PickConfig(objects=[CubeObject(color="red"), CubeObject(color="blue")])
    env = env_factory("warp", task="PickLift", config=config)
    import torch

    env.reset(seed=8, options={"target_index": target_index})
    _, _, _, _, info = env.step(torch.zeros(env.action_space.shape))
    expected = [1, 1] if np.ndim(target_index) == 0 else [1, 0]
    assert info["target_index"].tolist() == expected


@pytest.mark.parametrize("control_mode", ["pd_joint_pos", "pd_ee_delta_pose"])
@pytest.mark.parametrize("shape", [(), (6,), (1, 6), (2, 1)])
@pytest.mark.parametrize("has_previous_action", [False, True])
def test_warp_rejects_action_shape_before_mutating_state(
    env_factory, control_mode, shape, has_previous_action
):
    env = env_factory("warp", control_mode=control_mode)
    import torch

    env.reset(seed=8)
    valid = torch.zeros(env.action_space.shape)
    if has_previous_action:
        env.step(valid)
    raw = env.unwrapped
    state = ["qpos", "qvel", "ctrl", "_elapsed", "_prev_target", "_has_prev_action"]
    if raw._prev_action is not None:
        state.append("_prev_action")
    snapshots = {name: getattr(raw, name).clone() for name in state}
    with pytest.raises(ValueError, match="actions shape"):
        env.step(torch.zeros(shape))
    for name, expected in snapshots.items():
        assert torch.equal(getattr(raw, name), expected)
    if not has_previous_action:
        assert raw._prev_action is None
    _, _, _, _, info = env.step(valid)
    assert info["energy_norm"].shape == (2,)
    assert torch.equal(info["action_delta_norm"], torch.zeros(2))
