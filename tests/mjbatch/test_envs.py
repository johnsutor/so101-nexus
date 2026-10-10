"""CPU batch task parity and Gymnasium lifecycle contracts."""

import gymnasium as gym
import numpy as np
import pytest

from so101_nexus import ControlMode, JointPositions, PhysicsConfig, TouchConfig

pytest.importorskip("mjbatch")

_TASKS = [
    ("Touch", 1),
    ("LookAt", 1),
    ("Move", 1),
    ("PickLift", 1),
    ("PickReturn", 1),
    ("PickAndPlace", 1),
    ("PickAndPlace", 2),
    ("StackCube", 1),
]


@pytest.mark.parametrize("task,version", _TASKS)
@pytest.mark.parametrize(
    "control_mode",
    [
        "pd_joint_pos",
        "pd_joint_delta_pos",
        "pd_joint_target_delta_pos",
        "pd_ee_pose",
        "pd_ee_delta_pose",
    ],
)
def test_seeded_rollout_matches_mujoco(env_factory, task, version, control_mode: ControlMode):
    batch = env_factory(backend="mjbatch", task=task, version=version, control_mode=control_mode)
    reference = env_factory(task=task, version=version, control_mode=control_mode)
    obs, info = batch.reset(seed=[17, 23])
    expected, expected_info = reference.reset(seed=17)
    np.testing.assert_array_equal(obs[0], expected)
    assert info["success"][0] == expected_info["success"]
    batch.action_space.seed(4)
    for _ in range(3):
        actions = batch.action_space.sample()
        obs, rewards, terminated, truncated, info = batch.step(actions)
        expected, reward, term, trunc, expected_info = reference.step(actions[0])
        np.testing.assert_allclose(obs[0], expected, rtol=1e-6, atol=1e-7)
        assert rewards[0] == pytest.approx(reward, abs=1e-7)
        assert terminated[0] == term
        assert truncated[0] == trunc
        assert info["success"][0] == expected_info["success"]


@pytest.mark.parametrize("mode", list(gym.vector.AutoresetMode))
def test_episode_limit_and_partial_reset(env_factory, mode):
    env = env_factory(backend="mjbatch", max_episode_steps=1, autoreset_mode=mode)
    env.reset(seed=[11, 22])
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    _, _, _, truncated, info = env.step(action)
    assert truncated.all()
    if mode == gym.vector.AutoresetMode.SAME_STEP:
        assert info["_final_info"].all()
    elif mode == gym.vector.AutoresetMode.NEXT_STEP:
        _, reward, _, truncated, _ = env.step(action)
        assert not truncated.any()
        assert not reward.any()
    else:
        with pytest.raises(AssertionError):
            env.step(action)
    untouched = env.envs[1].unwrapped.data.qpos.copy()
    env.reset(seed=[33, None], options={"reset_mask": np.array([True, False])})
    np.testing.assert_array_equal(env.envs[1].unwrapped.data.qpos, untouched)


def test_physics_and_observation_config(env_factory):
    config = TouchConfig(observations=[JointPositions()], physics=PhysicsConfig(timestep_s=0.0025))
    env = env_factory(backend="mjbatch", config=config, num_threads=1)
    obs, _ = env.reset(seed=8)
    assert obs.shape == (2, 6)
    before = env.envs[0].unwrapped.data.time
    env.step(np.zeros(env.action_space.shape, dtype=np.float32))
    assert env.envs[0].unwrapped.data.time - before == pytest.approx(0.02)
    assert env.batch.num_threads == 1


@pytest.mark.parametrize("kwargs", [{"num_envs": 0}, {"num_threads": -1}])
def test_invalid_batch_size_or_threads(kwargs):
    import so101_nexus.mjbatch  # noqa: F401

    with pytest.raises(ValueError, match="num_envs must be positive"):
        gym.make_vec("MJBatchTouch-v1", **kwargs)


def test_object_pool_reset_and_native_execution(env_factory, monkeypatch):
    from so101_nexus import CubeObject

    config = TouchConfig(
        objects=[CubeObject(color="red"), CubeObject(color="blue")],
        robot_colors=["yellow", "blue"],
    )
    env = env_factory(backend="mjbatch", config=config)
    reference = env_factory(config=config)
    for target in (0, 1, 0):
        obs, _ = env.reset(seed=[17, 23], options={"target_index": target})
        expected, _ = reference.reset(seed=17, options={"target_index": target})
        np.testing.assert_array_equal(obs[0], expected)
        assert env.task_descriptions[0] == reference.unwrapped.task_description
        assert env.envs[0].unwrapped.config is not env.envs[1].unwrapped.config
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        obs, rewards, *_ = env.step(action)
        expected, reward, *_ = reference.step(action[0])
        np.testing.assert_allclose(obs[0], expected, atol=1e-7)
        assert rewards[0] == pytest.approx(reward)
    calls = []
    original = env.batch.step

    def record_step(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(env.batch, "step", record_step)
    env.step(action)
    assert len(calls) == 1
    assert calls[0]["nstep"] == env.envs[0].unwrapped._N_SUBSTEPS - 1


def test_camera_observations_and_render(env_factory):
    from so101_nexus import WristCamera

    config = TouchConfig(observations=[JointPositions(), WristCamera(width=32, height=32)])
    env = env_factory(backend="mjbatch", config=config, render_mode="rgb_array")
    obs, _ = env.reset(seed=4)
    assert obs["state"].shape == (2, 6)
    assert obs["wrist_camera"].shape == (2, 32, 32, 3)
    assert obs["wrist_camera"].dtype == np.uint8
    images = env.render()
    assert len(images) == 2
    assert all(image.ndim == 3 for image in images)


def test_envhub_preserves_native_batch_and_lerobot_contract():
    from so101_nexus.envhub import make_env

    env = make_env(env_id="MJBatchTouch-v1", n_envs=2, episode_length=1)["MJBatchTouch-v1"][0]
    try:
        obs, _ = env.reset(seed=8)
        assert obs["agent_pos"].shape == (2, 6)
        assert env.batch.num_sims == 2
        _, _, _, truncated, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        assert truncated.all()
        assert info["_final_info"].all()
        assert "is_success" in info["final_info"]
    finally:
        env.close()


def test_next_step_autoreset_does_not_advance_finished_world(env_factory):
    env = env_factory(backend="mjbatch", max_episode_steps=2)
    env.reset(seed=[3, 9])
    actions = np.zeros(env.action_space.shape, dtype=np.float32)
    env.step(actions)
    env.reset(options={"reset_mask": np.array([True, False])})
    _, _, _, truncated, _ = env.step(actions)
    np.testing.assert_array_equal(truncated, [False, True])
    _, reward, _, truncated, _ = env.step(actions)
    np.testing.assert_array_equal(truncated, [True, False])
    assert reward[1] == 0
    task = env.envs[1].unwrapped
    assert task.data.time == pytest.approx(task.config.reset_settle_frames * task.control_dt)


def test_one_substep_and_disabled_time_limit(env_factory):
    config = TouchConfig(physics=PhysicsConfig(timestep_s=0.02))
    env = env_factory(backend="mjbatch", config=config, max_episode_steps=-1)
    env.reset(seed=6)
    before = env.envs[0].unwrapped.data.time
    _, _, _, truncated, _ = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
    assert env.envs[0].unwrapped.data.time - before == pytest.approx(0.02)
    assert not truncated.any()
    env.close()
    env.close()
    assert not hasattr(env, "batch")
