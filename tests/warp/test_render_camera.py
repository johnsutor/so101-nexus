"""Batched visualization camera placement and rendering contracts."""

import gymnasium as gym
import numpy as np
import pytest
import torch

import so101_nexus.warp  # noqa: F401
from so101_nexus import LookAtConfig, RenderConfig
from so101_nexus.camera_utils import compute_angled_camera_params

pytestmark = pytest.mark.warp


def _side_camera_pose(env):
    forward = -env._cam_xmat[:, env._visual_cam_id, :, 2]
    azimuth = torch.rad2deg(torch.atan2(forward[:, 1], forward[:, 0])) % 360
    elevation = torch.rad2deg(torch.asin(forward[:, 2].clamp(-1, 1)))
    params = compute_angled_camera_params(
        spawn_center=env.config.spawn_center,
        spawn_max_radius=env.config.spawn_max_radius,
        spawn_angle_half_range_deg=env.config.spawn_angle_half_range_deg,
        spawn_half_size=(
            env.config.spawn_half_size if isinstance(env.config, LookAtConfig) else None
        ),
        aspect=env.config.render.width / env.config.render.height,
        azimuth=azimuth,
        elevation=elevation,
    )
    target = torch.as_tensor(params["lookat"], dtype=forward.dtype, device=env.device)
    distance = torch.linalg.vector_norm(target - env._cam_pos[:, env._visual_cam_id], dim=-1)
    return params, distance, azimuth, elevation, forward


@pytest.mark.parametrize("env_id", sorted(k for k in gym.registry if k.startswith("Warp")))
@pytest.mark.parametrize("mode", ["rgb_array", "depth_array"])
def test_render_all_tasks(env_factory, env_id, mode):
    task, version = env_id.removeprefix("Warp").split("-v")
    env = env_factory(backend="warp", task=task, version=int(version), render_mode=mode)
    env.config.render = RenderConfig(
        width=24,
        height=16,
        camera="side",
        side_azimuth_range_deg=(100.0, 200.0),
        side_elevation_range_deg=(-60.0, -20.0),
        side_distance_range=(0.8, 1.2),
    )
    assert env._visual_render_ctx is None
    obs, _ = env.reset(seed=42)
    assert env._visual_render_ctx is None
    assert isinstance(obs, torch.Tensor)
    pose = env._cam_pos[:, env._visual_cam_id].clone()
    quat = env._cam_quat[:, env._visual_cam_id].clone()
    assert not torch.equal(pose[0], pose[1])
    image = env.render()
    expected_shape = (2, 16, 24, 3) if mode == "rgb_array" else (2, 16, 24)
    assert image.shape == expected_shape
    assert image.dtype == (torch.uint8 if mode == "rgb_array" else torch.float32)
    assert image.device == env.device
    assert torch.isfinite(image).all()
    assert image.sum() > 0
    _, distance, azimuth, elevation, _ = _side_camera_pose(env)
    for value, low, high in ((distance, 0.8, 1.2), (azimuth, 100, 200), (elevation, -60, -20)):
        assert ((value >= low) & (value <= high)).all()
    snapshot = image.clone()
    env.render()
    torch.testing.assert_close(image, snapshot)
    env.reset(seed=42)
    torch.testing.assert_close(env._cam_pos[:, env._visual_cam_id], pose)
    torch.testing.assert_close(env._cam_quat[:, env._visual_cam_id], quat)
    torch.testing.assert_close(env.render(), snapshot)
    env._write_reset_state(torch.tensor([True, False], device=env.device))
    assert not torch.equal(env._cam_pos[0, env._visual_cam_id], pose[0])
    torch.testing.assert_close(env._cam_pos[1, env._visual_cam_id], pose[1])


def test_render_disabled_is_noop(env_factory):
    env = env_factory(backend="warp")
    env.reset(seed=0)
    assert env.render() is None
    assert env._visual_cam_id == -1


@pytest.mark.parametrize("camera", ["side", "overhead"])
@pytest.mark.parametrize("mode", ["rgb_array", "depth_array"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_fixed_pose_matches_mujoco(env_factory, camera, device, mode):
    import mujoco

    from so101_nexus import TouchConfig

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    render = RenderConfig(
        width=24,
        height=16,
        camera=camera,
        side_azimuth_range_deg=(123.0, 123.0),
        side_elevation_range_deg=(-45.0, -45.0),
        side_distance_range=(0.9, 0.9),
    )
    config = TouchConfig(render=render, terminate_on_success=False)
    warp_env = env_factory(backend="warp", config=config, device=device, render_mode=mode)
    native = env_factory(config=config)
    native.reset(seed=0)
    warp_env.reset(seed=0)
    params = native.unwrapped._render_camera_params()
    cam = mujoco.MjvCamera()
    for name, value in params.items():
        setattr(cam, name, value)
    scene = mujoco.MjvScene(native.unwrapped.model, maxgeom=1000)
    mujoco.mjv_updateScene(
        native.unwrapped.model,
        native.unwrapped.data,
        mujoco.MjvOption(),
        None,
        cam,
        mujoco.mjtCatBit.mjCAT_ALL,
        scene,
    )
    center = np.mean([c.pos for c in scene.camera], axis=0)
    cid = warp_env._visual_cam_id
    np.testing.assert_allclose(warp_env._cam_xpos[:, cid].cpu(), np.tile(center, (2, 1)), atol=1e-6)
    np.testing.assert_allclose(
        -warp_env._cam_xmat[:, cid, :, 2].cpu(), np.tile(scene.camera[0].forward, (2, 1)), atol=1e-6
    )
    np.testing.assert_allclose(
        warp_env._cam_xmat[:, cid, :, 1].cpu(), np.tile(scene.camera[0].up, (2, 1)), atol=1e-6
    )
    first = warp_env.render().clone()
    assert first.sum() > 0
    pose = warp_env._cam_pos[:, cid].clone()
    warp_env.step(warp_env._joint_qpos().clone())
    torch.testing.assert_close(warp_env._cam_pos[:, cid], pose)
    torch.testing.assert_close(warp_env.render(), warp_env.render())


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_visualization_coexists_with_observations_and_autoreset(env_factory, device):
    from so101_nexus import JointPositions, OverheadCamera, TouchConfig, WristCamera

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    config = TouchConfig(
        render=RenderConfig(width=24, height=16, camera="side", side_distance_range=(0.8, 1.2)),
        observations=[
            JointPositions(),
            WristCamera(width=16, height=12),
            OverheadCamera(width=16, height=12),
        ],
        terminate_on_success=False,
    )
    env = env_factory(
        backend="warp", device=device, config=config, render_mode="rgb_array", max_episode_steps=2
    )
    obs, _ = env.reset(seed=5)
    assert set(obs) == {"state", "wrist_camera", "overhead_camera"}
    pose = env._cam_pos[:, env._visual_cam_id].clone()
    old = env.render()
    snapshot = old.clone()
    env._elapsed[0] = 1
    _, _, _, truncated, _ = env.step(env._joint_qpos().clone())
    assert truncated.tolist() == [True, False]
    assert not torch.equal(env._cam_pos[0, env._visual_cam_id], pose[0])
    torch.testing.assert_close(env._cam_pos[1, env._visual_cam_id], pose[1])
    env.render()
    torch.testing.assert_close(old, snapshot)


@pytest.mark.parametrize("half_range", [30.0, 180.0])
@pytest.mark.parametrize("distance_range", [None, (0.9, 0.9), (0.8, 1.2)])
def test_random_angles_refit_target_and_distance_to_spawn_arc(
    env_factory, half_range, distance_range
):
    env = env_factory(backend="warp", render_mode="rgb_array")
    env.config.spawn_angle_half_range_deg = half_range
    env.config.render = RenderConfig(
        camera="side",
        side_azimuth_range_deg=(120.0, 200.0),
        side_elevation_range_deg=(-45.0, -20.0),
        side_distance_range=distance_range,
    )
    env.reset(seed=42)
    params, distance, azimuth, elevation, forward = _side_camera_pose(env)
    assert isinstance(params["lookat"], torch.Tensor)
    assert params["lookat"].shape == (env.num_envs, 3)
    assert not torch.equal(params["lookat"][0], params["lookat"][1])
    if distance_range is None:
        torch.testing.assert_close(distance, params["distance"])
    else:
        low, high = distance_range
        assert ((distance >= low - 1e-6) & (distance <= high + 1e-6)).all()
    for index in range(env.num_envs):
        expected = compute_angled_camera_params(
            spawn_center=env.config.spawn_center,
            spawn_max_radius=env.config.spawn_max_radius,
            spawn_angle_half_range_deg=half_range,
            aspect=env.config.render.width / env.config.render.height,
            azimuth=float(azimuth[index]),
            elevation=float(elevation[index]),
        )
        expected_distance = (
            expected["distance"] if distance_range is None else float(distance[index])
        )
        eye = expected["lookat"] - expected_distance * forward[index].cpu().numpy()
        np.testing.assert_allclose(env._cam_pos[index, env._visual_cam_id].cpu(), eye, atol=1e-6)


def test_look_at_side_camera_fits_square_spawn(env_factory):
    config = LookAtConfig(
        spawn_half_size=0.4,
        spawn_max_radius=0.4,
        render=RenderConfig(
            camera="side",
            side_azimuth_range_deg=(200.0, 200.0),
            side_elevation_range_deg=(-20.0, -20.0),
        ),
    )
    env = env_factory(backend="warp", task="LookAt", config=config, render_mode="rgb_array")
    env.reset(seed=0)
    expected = compute_angled_camera_params(
        spawn_center=config.spawn_center,
        spawn_max_radius=config.spawn_max_radius,
        spawn_half_size=config.spawn_half_size,
        azimuth=200,
        elevation=-20,
    )
    target = torch.as_tensor(expected["lookat"], dtype=torch.float32, device=env.device)
    distance = torch.linalg.vector_norm(target - env._cam_pos[:, env._visual_cam_id], dim=-1)
    torch.testing.assert_close(distance, torch.full_like(distance, float(expected["distance"])))
