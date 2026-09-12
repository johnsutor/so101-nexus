"""Shared PickReturn runtime contracts on both physics backends."""

import numpy as np
import pytest

from so101_nexus import (
    CubeObject,
    PickReturnConfig,
    RestingJointPositions,
    RobotConfig,
    TargetOffset,
    TargetPosition,
    component_slice,
)


@pytest.fixture(params=["mujoco", "warp"])
def backend(request):
    return request.param


def as_numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def test_registration_default_reset_and_step(backend, env_factory):
    env = env_factory(backend=backend, task="PickReturn")
    obs, _ = env.reset(seed=3)
    assert obs.shape[-1] == 42
    config = env.unwrapped.config
    rest = as_numpy(obs)[..., component_slice(config.observations, RestingJointPositions)]
    np.testing.assert_allclose(rest, np.broadcast_to(config.robot.rest_qpos_deg[:5], rest.shape))
    assert "return the arm to rest" in env.unwrapped.task_description
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert np.isfinite(as_numpy(obs)).all()
    assert np.isfinite(as_numpy(reward)).all()
    assert not as_numpy(terminated).any()
    assert not as_numpy(truncated).any()
    np.testing.assert_allclose(sum(info["reward_components"].values()), reward, atol=1e-6)


def test_custom_rest_target_is_independent_of_reset_pose(backend, env_factory):
    robot = RobotConfig(rest_qpos_deg=(25.0, -75.0, 80.0, 30.0, 15.0, 0.0))
    config = PickReturnConfig(robot=robot, reset_settle_frames=0)
    env = env_factory(backend=backend, task="PickReturn", config=config)
    obs, _ = env.reset(seed=2, options={"init_qpos": np.zeros(6)})
    obs = as_numpy(obs)
    rest = obs[..., component_slice(config.observations, RestingJointPositions)]
    np.testing.assert_allclose(rest, np.broadcast_to(robot.rest_qpos_deg[:5], rest.shape))
    goal = obs[..., component_slice(config.observations, TargetPosition)]
    offset = obs[..., component_slice(config.observations, TargetOffset)]
    assert np.linalg.norm(offset) > 0.05
    after, _ = env.reset(seed=9, options={"init_qpos": np.asarray(robot.rest_qpos_rad)})
    after = as_numpy(after)
    np.testing.assert_allclose(
        after[..., component_slice(config.observations, TargetPosition)], goal
    )
    np.testing.assert_allclose(
        after[..., component_slice(config.observations, TargetOffset)], 0, atol=1e-6
    )


def test_target_pool_description_tracks_selected_object(backend, env_factory):
    from so101_nexus import CylinderObject

    objects = [CubeObject(color="red"), CylinderObject(color="blue")]
    env = env_factory(
        backend=backend,
        task="PickReturn",
        config=PickReturnConfig(objects=objects, n_distractors=1),
    )
    env.reset(seed=0, options={"target_index": 1})
    assert env.unwrapped.task_description == env.unwrapped.config.describe_target(objects[1])


def test_every_state_component_is_routed(backend, env_factory):
    from so101_nexus.observations import _state_component_types

    components = [cls() for cls in _state_component_types().values()]
    config = PickReturnConfig(observations=components)
    env = env_factory(backend=backend, task="PickReturn", config=config)
    for obs in (env.reset(seed=0)[0], env.step(env.action_space.sample())[0]):
        assert obs.shape[-1] == sum(c.size for c in components)
        assert np.isfinite(as_numpy(obs)).all()


@pytest.mark.parametrize("angles", [(180, 0, 0, 0, 0, 0), (0, 0, 120, 0, 0, 0)])
def test_unreachable_rest_target_rejected(backend, env_factory, angles):
    with pytest.raises(ValueError, match="rest_qpos_deg"):
        env_factory(
            backend=backend,
            task="PickReturn",
            config=PickReturnConfig(robot=RobotConfig(rest_qpos_deg=angles)),
        )


def set_return_state(env, backend, monkeypatch, *, error=0.0, height=0.06, speed=0.0, grasped=True):
    """Set joint/object geometry while isolating the independently tested grasp detector."""
    e = env.unwrapped
    if backend == "warp":
        import torch

        joints = torch.tensor(e.config.robot.rest_qpos_rad, dtype=torch.float32)
        joints[-1] = 0.7
        joints[0] += np.radians(error)
        e.qpos[:, e._qpos_adr] = joints
        e.qvel[:, e._dof_adr] = speed
        e.qvel[:, e._dof_adr[-1]] = 10.0
        e.qpos[e._world_rows, e._target_qadr + 2] = e._initial_obj_z + height
        monkeypatch.setattr(e, "_is_grasping", lambda: torch.full((2,), float(grasped)))
    else:
        joints = np.asarray(e.config.robot.rest_qpos_rad)
        joints[-1] = 0.7
        joints[0] += np.radians(error)
        e.data.qpos[e._qpos_addrs] = joints
        e.data.qvel[e._qvel_addrs] = speed
        e.data.qvel[e._qvel_addrs[-1]] = 10.0
        e.data.qpos[e._slots[e._target_slot_idx].qpos_addr + 2] = e._initial_obj_z + height
        monkeypatch.setattr(e, "_is_grasping", lambda: float(grasped))


@pytest.mark.parametrize(
    ("error", "height", "speed", "grasped", "expected"),
    [
        (0, 0.06, 0, True, True),
        (10, 0.06, 0, True, False),
        (0, 0.01, 0, True, False),
        (0, 0.06, 0.3, True, False),
        (0, 0.06, 0, False, False),
    ],
)
def test_success_requires_every_condition_and_ignores_gripper(
    backend,
    env_factory,
    monkeypatch,
    error,
    height,
    speed,
    grasped,
    expected,
):
    env = env_factory(backend=backend, task="PickReturn")
    env.reset(seed=1)
    monkeypatch.setattr(env.unwrapped, "_advance_physics", lambda: None)
    set_return_state(
        env, backend, monkeypatch, error=error, height=height, speed=speed, grasped=grasped
    )
    _, _, terminated, _, info = env.step(env.action_space.sample())
    assert np.all(as_numpy(info["success"]) == expected)
    assert np.all(as_numpy(terminated) == expected)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"return_threshold_deg": 15},
        {"lift_threshold": 0.005},
        {"robot": RobotConfig(static_vel_threshold=0.5)},
    ],
)
def test_nondefault_thresholds_change_runtime_success(backend, env_factory, monkeypatch, kwargs):
    cfg = PickReturnConfig(**kwargs)
    env = env_factory(backend=backend, task="PickReturn", config=cfg)
    env.reset(seed=1)
    state = (
        {"error": 10}
        if "return_threshold_deg" in kwargs
        else ({"height": 0.01} if "lift_threshold" in kwargs else {"speed": 0.3})
    )
    set_return_state(env, backend, monkeypatch, **state)
    monkeypatch.setattr(env.unwrapped, "_advance_physics", lambda: None)
    _, _, terminated, _, info = env.step(env.action_space.sample())
    assert as_numpy(info["success"]).all()
    assert as_numpy(terminated).all()


def test_reward_dwell_cycles_and_success_override(backend, env_factory, monkeypatch):
    cfg = PickReturnConfig(terminate_on_success=False)
    env = env_factory(backend=backend, task="PickReturn", config=cfg)
    env.reset(seed=1)
    monkeypatch.setattr(env.unwrapped, "_advance_physics", lambda: None)
    action = env.action_space.sample()
    set_return_state(env, backend, monkeypatch, error=30)
    env.step(action)
    np.testing.assert_allclose(env.step(action)[1], 0, atol=1e-6)
    set_return_state(env, backend, monkeypatch, error=30, grasped=False)
    lost = as_numpy(env.step(action)[1])
    assert (lost < 0).all()
    set_return_state(env, backend, monkeypatch, error=30)
    recovered = as_numpy(env.step(action)[1])
    np.testing.assert_allclose(lost + recovered, 0, atol=1e-6)
    set_return_state(env, backend, monkeypatch)
    _, reward, terminated, _, info = env.step(action)
    assert as_numpy(info["success"]).all()
    assert not as_numpy(terminated).any()
    np.testing.assert_allclose(reward, 1.0)


def test_lift_shaping_config_affects_runtime_reward(backend, env_factory, monkeypatch):
    rewards = []
    for height_scale in (0.04, 0.16):
        env = env_factory(
            backend=backend,
            task="PickReturn",
            config=PickReturnConfig(max_goal_height=height_scale),
        )
        env.reset(seed=1)
        monkeypatch.setattr(env.unwrapped, "_advance_physics", lambda: None)
        action = env.action_space.sample()
        set_return_state(env, backend, monkeypatch, height=0, error=90)
        env.step(action)
        set_return_state(env, backend, monkeypatch, height=0.025, error=90)
        rewards.append(as_numpy(env.step(action)[1]))
    assert (rewards[0] > rewards[1]).all()


def test_reset_potentials_match_returned_state(backend, env_factory, monkeypatch):
    env = env_factory(
        backend=backend, task="PickReturn", config=PickReturnConfig(reset_settle_frames=5)
    )
    for seed in (1, 2):
        env.reset(seed=seed)
        monkeypatch.setattr(env.unwrapped, "_advance_physics", lambda: None)
        _, reward, _, _, _ = env.step(env.action_space.sample())
        np.testing.assert_allclose(reward, 0.0, atol=1e-6)


@pytest.mark.parametrize("obs_mode", ["state", "visual"])
def test_rgb_depth_and_state_modes(backend, env_factory, obs_mode):
    from so101_nexus import OverheadCamera, WristCamera

    components = [
        RestingJointPositions(),
        TargetOffset(),
        WristCamera(width=24, height=16, modalities=("rgb", "depth")),
        OverheadCamera(width=20, height=12, modalities=("rgb", "depth")),
    ]
    env = env_factory(
        backend=backend,
        task="PickReturn",
        config=PickReturnConfig(observations=components, obs_mode=obs_mode),
    )
    for result in (
        env.reset(seed=0),
        env.step(env.action_space.sample()),
    ):
        obs, info = result[0], result[-1]
        assert set(obs) == {
            "state",
            "wrist_camera",
            "wrist_camera_depth",
            "overhead_camera",
            "overhead_camera_depth",
        }
        assert obs["state"].shape[-1] == (6 if obs_mode == "visual" else 8)
        if obs_mode == "visual":
            assert info["privileged_state"].shape[-1] == 8
        for camera, height, width in (("wrist_camera", 16, 24), ("overhead_camera", 12, 20)):
            assert obs[camera].shape[-3:] == (height, width, 3)
            assert obs[camera + "_depth"].shape[-2:] == (height, width)
            assert as_numpy(obs[camera]).dtype == np.uint8
            assert as_numpy(obs[camera + "_depth"]).dtype == np.float32
            assert (as_numpy(obs[camera + "_depth"]) >= 0).all()


def test_cross_backend_reward_parity(env_factory, monkeypatch):
    import torch

    envs = {
        backend: env_factory(backend=backend, task="PickReturn") for backend in ("mujoco", "warp")
    }
    m, w = envs["mujoco"].unwrapped, envs["warp"]
    monkeypatch.setattr(
        m,
        "_get_tcp_pose",
        lambda: np.r_[m._get_target_pose()[:3] + np.array([0.02, 0, 0]), 1, 0, 0, 0],
    )
    monkeypatch.setattr(w, "_tcp_pos", lambda: w._target_pos() + torch.tensor([0.02, 0, 0]))
    for backend, env in envs.items():
        env.reset(seed=0)
        monkeypatch.setattr(env.unwrapped, "_advance_physics", lambda: None)
        set_return_state(env, backend, monkeypatch, height=0, error=90, grasped=True)
        env.step(env.action_space.sample())
    for state in (
        {"height": 0.02, "error": 90},
        {"error": 90},
        {"error": 30},
        {"error": 30},
        {"error": 30, "grasped": False},
        {"error": 30},
        {"error": 3, "speed": 0.4},
        {"error": 3},
    ):
        results = []
        for backend, env in envs.items():
            set_return_state(env, backend, monkeypatch, **state)
            results.append(env.step(env.action_space.sample()))
        scalar, batched = results
        np.testing.assert_allclose(batched[1], scalar[1], atol=1e-6)
        for name, value in scalar[-1]["reward_components"].items():
            np.testing.assert_allclose(batched[-1]["reward_components"][name], value, atol=1e-6)
        assert np.all(as_numpy(batched[2]) == scalar[2])


def test_warp_partial_autoreset_preserves_ongoing_world(env_factory, monkeypatch):
    import torch

    env = env_factory(
        backend="warp", task="PickReturn", config=PickReturnConfig(reset_settle_frames=0)
    )
    env.reset(seed=1)
    set_return_state(env, "warp", monkeypatch, error=30)
    monkeypatch.setattr(env, "_advance_physics", lambda: None)
    monkeypatch.setattr(env, "_is_grasping", lambda: torch.tensor([0.0, 1.0]))
    action = env.action_space.sample()
    env.step(action)
    ongoing = [value[1].clone() for value in env._return_previous]
    env._elapsed[0] = env.max_episode_steps - 1
    _, _, terminated, truncated, info = env.step(action)
    assert terminated.tolist() == [False, False]
    assert truncated.tolist() == [True, False]
    assert env._elapsed.tolist() == [0, 2]
    for prev, expected in zip(env._return_previous, ongoing, strict=True):
        assert prev[1] == expected
    assert info["reward_components"]["task_objective"][1] == 0
    np.testing.assert_allclose(env.step(action)[1], 0, atol=1e-6)


@pytest.mark.parametrize(
    "control_mode",
    ["pd_joint_delta_pos", "pd_joint_target_delta_pos", "pd_ee_pose", "pd_ee_delta_pose"],
)
def test_all_control_modes(backend, env_factory, control_mode):
    env = env_factory(backend=backend, task="PickReturn", control_mode=control_mode)
    env.reset(seed=1)
    obs, reward, _, _, _ = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
    assert np.isfinite(as_numpy(obs)).all()
    assert np.isfinite(as_numpy(reward)).all()
