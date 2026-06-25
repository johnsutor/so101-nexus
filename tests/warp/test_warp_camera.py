"""Camera observation tests for the MuJoCo Warp backend (Warp CPU device)."""

import pytest

pytestmark = pytest.mark.warp

# Tiny resolutions keep the CPU ray-trace fast while still exercising H != W.
WRIST_W, WRIST_H = 24, 16
OVER_W, OVER_H = 20, 12


def _touch_env(observations, *, num_envs=2, seed=0, obs_mode="state", max_episode_steps=512):
    from so101_nexus.config import TouchConfig
    from so101_nexus.warp.touch_env import WarpTouchVectorEnv

    config = TouchConfig(observations=observations, obs_mode=obs_mode)
    return WarpTouchVectorEnv(
        num_envs=num_envs,
        config=config,
        device="cpu",
        seed=seed,
        max_episode_steps=max_episode_steps,
    )


def test_wrist_camera_obs_shape_and_dtype():
    import torch
    from gymnasium import spaces

    from so101_nexus.observations import JointPositions, WristCamera

    env = _touch_env([JointPositions(), WristCamera(width=WRIST_W, height=WRIST_H)])
    obs, _ = env.reset(seed=0)
    assert isinstance(obs, dict)
    assert set(obs) == {"state", "wrist_camera"}
    assert obs["state"].shape == (2, 6)
    img = obs["wrist_camera"]
    assert img.shape == (2, WRIST_H, WRIST_W, 3)
    assert img.dtype == torch.uint8
    assert int(img.max()) <= 255
    assert int(img.min()) >= 0
    assert int(img.sum()) > 0  # not all black
    assert isinstance(env.single_observation_space, spaces.Dict)
    assert env.single_observation_space["wrist_camera"].shape == (WRIST_H, WRIST_W, 3)


def test_overhead_only_uses_precomputed_rays_and_renders():
    import torch

    from so101_nexus.observations import JointPositions, OverheadCamera

    env = _touch_env([JointPositions(), OverheadCamera(width=OVER_W, height=OVER_H)])
    # No wrist camera -> no per-world reallocation, precomputed rays.
    assert env._wrist_cam is None
    assert env._render_ctx.use_precomputed_rays is True
    obs, _ = env.reset(seed=0)
    assert set(obs) == {"state", "overhead_camera"}
    img = obs["overhead_camera"]
    assert img.shape == (2, OVER_H, OVER_W, 3)
    assert img.dtype == torch.uint8
    assert int(img.sum()) > 0


def test_both_cameras_distinct_resolutions():
    from so101_nexus.observations import JointPositions, OverheadCamera, WristCamera

    env = _touch_env(
        [
            JointPositions(),
            WristCamera(width=WRIST_W, height=WRIST_H),
            OverheadCamera(width=OVER_W, height=OVER_H),
        ]
    )
    obs, _ = env.reset(seed=0)
    assert set(obs) == {"state", "wrist_camera", "overhead_camera"}
    assert obs["wrist_camera"].shape == (2, WRIST_H, WRIST_W, 3)
    assert obs["overhead_camera"].shape == (2, OVER_H, OVER_W, 3)
    # Wrist present -> per-world rays recomputed for FOV randomization.
    assert env._render_ctx.use_precomputed_rays is False


def test_state_only_returns_flat_tensor():
    import torch

    from so101_nexus.observations import JointPositions, ObjectOffset

    env = _touch_env([JointPositions(), ObjectOffset()])
    obs, _ = env.reset(seed=0)
    assert isinstance(obs, torch.Tensor)
    assert obs.shape == (2, 9)
    assert env._has_cameras is False


def _visual_pick_env(*, num_envs=2, seed=0, max_episode_steps=512):
    from so101_nexus.config import PickConfig
    from so101_nexus.observations import EndEffectorPose, JointPositions, WristCamera
    from so101_nexus.warp.pick_env import WarpPickLiftVectorEnv

    config = PickConfig(
        obs_mode="visual",
        observations=[
            JointPositions(),
            EndEffectorPose(),
            WristCamera(width=WRIST_W, height=WRIST_H),
        ],
    )
    return WarpPickLiftVectorEnv(
        num_envs=num_envs,
        config=config,
        device="cpu",
        seed=seed,
        max_episode_steps=max_episode_steps,
    )


def test_visual_mode_state_is_joints_and_privileged_state_in_info():
    env = _visual_pick_env()
    obs, info = env.reset(seed=0)
    # visual "state" is joint positions only; full state (6 + 7 = 13) is privileged.
    assert obs["state"].shape == (2, 6)
    assert "privileged_state" in info
    assert info["privileged_state"].shape == (2, 13)
    assert obs["wrist_camera"].shape == (2, WRIST_H, WRIST_W, 3)


def test_same_step_autoreset_returns_post_reset_visual_obs():
    import torch

    env = _visual_pick_env(max_episode_steps=1)
    env.reset(seed=0)
    obs, _, _, truncated, info = env.step(torch.zeros((2, 6)))
    assert bool(truncated.all())  # max_episode_steps=1 truncates every world
    # Post-autoreset observation still carries images and privileged state.
    assert obs["state"].shape == (2, 6)
    assert obs["wrist_camera"].shape == (2, WRIST_H, WRIST_W, 3)
    assert info["privileged_state"].shape == (2, 13)


def test_wrist_dr_is_seeded_deterministic_and_seed_sensitive():
    import torch

    from so101_nexus.observations import JointPositions, WristCamera

    obs_a, _ = _touch_env(
        [JointPositions(), WristCamera(width=WRIST_W, height=WRIST_H)], seed=11
    ).reset(seed=11)
    obs_b, _ = _touch_env(
        [JointPositions(), WristCamera(width=WRIST_W, height=WRIST_H)], seed=11
    ).reset(seed=11)
    obs_c, _ = _touch_env(
        [JointPositions(), WristCamera(width=WRIST_W, height=WRIST_H)], seed=99
    ).reset(seed=99)
    assert torch.equal(obs_a["wrist_camera"], obs_b["wrist_camera"])
    assert not torch.equal(obs_a["wrist_camera"], obs_c["wrist_camera"])


def test_wrist_dr_writes_wxyz_pitch_quaternion():
    import numpy as np
    import torch

    from so101_nexus.observations import JointPositions, WristCamera

    pitch_deg = -20.0
    env = _touch_env(
        [
            JointPositions(),
            WristCamera(
                width=WRIST_W,
                height=WRIST_H,
                pitch_deg_range=(pitch_deg, pitch_deg),
                fov_deg_range=(60.0, 60.0),
                pos_x_noise=0.0,
                pos_y_noise=0.0,
                pos_z_noise=0.0,
            ),
        ]
    )
    env.reset(seed=0)
    quat = env._cam_quat[:, env._wrist_mjid]  # (N, 4) wxyz
    half = np.radians(pitch_deg) / 2.0
    expected = torch.tensor([np.cos(half), np.sin(half), 0.0, 0.0], dtype=quat.dtype)
    assert torch.allclose(quat[0], expected, atol=1e-5)
    # Degenerate FOV range -> every world shares the same pinned fovy.
    fovy = env._cam_fovy[:, env._wrist_mjid]
    assert torch.allclose(fovy, torch.full_like(fovy, 60.0), atol=1e-4)


def test_wrist_dr_fovy_varies_per_world():
    from so101_nexus.observations import JointPositions, WristCamera

    env = _touch_env(
        [JointPositions(), WristCamera(width=WRIST_W, height=WRIST_H, fov_deg_range=(50.0, 90.0))],
        num_envs=8,
    )
    env.reset(seed=0)
    fovy = env._cam_fovy[:, env._wrist_mjid]
    assert float(fovy.std()) > 0.0  # per-world FOV randomization is live


def test_lookat_uses_live_per_world_fov_for_success():
    import torch

    from so101_nexus.config import LookAtConfig
    from so101_nexus.observations import GazeDirection, WristCamera
    from so101_nexus.warp.look_at_env import WarpLookAtVectorEnv

    env = WarpLookAtVectorEnv(
        num_envs=2,
        config=LookAtConfig(
            observations=[GazeDirection(), WristCamera(width=WRIST_W, height=WRIST_H)]
        ),
        device="cpu",
        seed=0,
    )
    env.reset(seed=0)
    # Force a degenerate per-world FOV: world 0 can never satisfy (0 deg), world 1
    # always satisfies (360 deg -> half-FOV = pi >= any orientation error). A scalar
    # boundary would make both worlds behave identically; per-world fovy must not.
    env._cam_fovy[:, env._wrist_mjid] = torch.tensor([0.0, 360.0])
    _, _, terminated, _, _ = env.step(torch.zeros((2, 6)))
    assert not bool(terminated[0])
    assert bool(terminated[1])


def test_bare_camera_observation_rejected():
    from so101_nexus.config import TouchConfig
    from so101_nexus.observations import CameraObservation, JointPositions
    from so101_nexus.warp.touch_env import WarpTouchVectorEnv

    # The abstract base camera class is not routed by _setup_cameras; reject it
    # fast instead of silently dropping it from the observation.
    config = TouchConfig(observations=[JointPositions(), CameraObservation(16, 16)])
    with pytest.raises(NotImplementedError):
        WarpTouchVectorEnv(num_envs=2, config=config, device="cpu")


def test_camera_only_config_has_empty_state():
    from so101_nexus.observations import WristCamera

    env = _touch_env([WristCamera(width=WRIST_W, height=WRIST_H)])
    obs, _ = env.reset(seed=0)
    assert obs["state"].shape == (2, 0)
    assert obs["wrist_camera"].shape == (2, WRIST_H, WRIST_W, 3)


def test_lookat_renders_target_marker():
    import torch

    from so101_nexus.config import LookAtConfig
    from so101_nexus.objects import CubeObject
    from so101_nexus.observations import JointPositions, OverheadCamera
    from so101_nexus.warp.look_at_env import WarpLookAtVectorEnv

    cfg = LookAtConfig(
        objects=[CubeObject(color="blue", half_size=0.03)],
        observations=[JointPositions(), OverheadCamera(width=48, height=48)],
    )
    env = WarpLookAtVectorEnv(num_envs=2, config=cfg, device="cpu", seed=0)
    obs, _ = env.reset(seed=0)
    # The visual-only marker geom tracks the per-world target tensor.
    assert torch.equal(env._geom_xpos[:, env._marker_gid], env._targets)
    # The blue target box is visible from overhead (distinct from gray floor and
    # yellow robot), proving the marker is actually rendered.
    img = obs["overhead_camera"][0]
    blue = int(((img[..., 2] > 120) & (img[..., 0] < 90) & (img[..., 1] < 90)).sum())
    assert blue > 0


def test_move_renders_target_marker():
    import torch

    from so101_nexus.config import MoveConfig
    from so101_nexus.observations import JointPositions, OverheadCamera
    from so101_nexus.warp.move_env import WarpMoveVectorEnv

    cfg = MoveConfig(observations=[JointPositions(), OverheadCamera(width=48, height=48)])
    env = WarpMoveVectorEnv(num_envs=2, config=cfg, device="cpu", seed=0)
    env.reset(seed=0)
    # The visual-only marker geom tracks the per-world target tensor.
    assert torch.equal(env._geom_xpos[:, env._marker_gid], env._targets)
