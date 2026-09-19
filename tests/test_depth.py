"""Metric camera observations on both simulation backends."""

import numpy as np
import pytest

from so101_nexus import (
    JointPositions,
    MoveConfig,
    OverheadCamera,
    RenderConfig,
    TouchConfig,
    WristCamera,
)


@pytest.fixture(params=["mujoco", pytest.param("warp", marks=pytest.mark.warp)])
def backend(request):
    return request.param


def _numpy(value):
    return value if isinstance(value, np.ndarray) else value.cpu().numpy()


@pytest.mark.parametrize("camera_type", [WristCamera, OverheadCamera])
@pytest.mark.parametrize("modalities", [(), ("segmentation",), ("depth", "depth"), "depth"])
def test_invalid_modalities(camera_type, modalities):
    with pytest.raises(ValueError, match="modalities"):
        camera_type(modalities=modalities)


@pytest.mark.parametrize(
    "modalities, overhead_modalities",
    [
        (("rgb",), ("depth",)),
        (("depth",), ("rgb",)),
        (("depth",), ("depth",)),
        (("rgb", "depth"), ("rgb", "depth")),
    ],
)
@pytest.mark.parametrize("obs_mode", ["state", "visual"])
def test_depth_observation_contract(
    env_factory, backend, modalities, overhead_modalities, obs_mode
):
    cameras = [
        WristCamera(width=24, height=16, modalities=modalities),
        OverheadCamera(width=20, height=12, modalities=overhead_modalities),
    ]
    env = env_factory(
        backend, config=TouchConfig(observations=[JointPositions(), *cameras], obs_mode=obs_mode)
    )
    obs, info = env.reset(seed=17)
    expected = {"state"}
    if "rgb" in overhead_modalities:
        expected.add("overhead_camera")
    if "depth" in overhead_modalities:
        expected.add("overhead_camera_depth")
    if "rgb" in modalities:
        expected.add("wrist_camera")
    if "depth" in modalities:
        expected.add("wrist_camera_depth")
    assert set(obs) == expected
    space = env.single_observation_space if backend == "warp" else env.observation_space
    batch = (2,) if backend == "warp" else ()
    saved = {key: _numpy(value).copy() for key, value in obs.items()}
    for key, value in saved.items():
        if key == "state":
            continue
        height, width = (16, 24) if key.startswith("wrist") else (12, 20)
        is_depth = key.endswith("_depth")
        shape = (height, width) if is_depth else (height, width, 3)
        assert value.shape == (*batch, *shape)
        assert value.dtype == (np.float32 if is_depth else np.uint8)
        assert space[key].contains(value[0] if batch else value)
        assert np.isfinite(value).all()
        if is_depth:
            assert (value > 0).all()
            assert np.ptp(value) > 0
    if obs_mode == "visual":
        assert "privileged_state" in info
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    if backend == "warp":
        import torch

        action = torch.from_numpy(action)
    stepped, *_ = env.step(action)
    assert set(stepped) == expected
    env.reset(seed=18)
    for key in obs:
        np.testing.assert_array_equal(_numpy(obs[key]), saved[key])


@pytest.mark.parametrize("camera", ["overhead", "side"])
def test_depth_array_render(env_factory, camera):
    env = env_factory(
        config=TouchConfig(render=RenderConfig(width=32, height=24, camera=camera)),
        render_mode="depth_array",
    )
    env.reset(seed=0)
    depth = env.render()
    assert "depth_array" in env.metadata["render_modes"]
    assert depth.shape == (24, 32)
    assert depth.dtype == np.float32
    assert np.isfinite(depth).all()
    assert (depth > 0).all()
    assert np.ptp(depth) > 0


def test_mujoco_rgb_and_depth_replay_after_dirty_reset(env_factory):
    cameras = [
        WristCamera(width=24, height=16, modalities=("rgb", "depth")),
        OverheadCamera(width=20, height=12, modalities=("rgb", "depth")),
    ]
    env = env_factory(config=TouchConfig(observations=[JointPositions(), *cameras]))

    expected, _ = env.reset(seed=29)
    action = np.full(env.action_space.shape, 0.25, dtype=np.float32)
    expected_step = env.step(action)[0]

    env.reset(seed=97)
    for _ in range(5):
        env.step(-action)

    actual, _ = env.reset(seed=29)
    actual_step = env.step(action)[0]
    for key in (
        "wrist_camera",
        "wrist_camera_depth",
        "overhead_camera",
        "overhead_camera_depth",
    ):
        np.testing.assert_array_equal(actual[key], expected[key])
        np.testing.assert_array_equal(actual_step[key], expected_step[key])


@pytest.mark.parametrize("view", ["floor", "sky", "beyond_far"])
def test_depth_matches_known_plane_distance(env_factory, backend, view):
    env = env_factory(
        backend,
        task="Move",
        config=MoveConfig(
            observations=[OverheadCamera(width=24, height=16, modalities=("depth",))]
        ),
    )
    initial, _ = env.reset(seed=0)
    assert initial["state"].shape == ((2, 0) if backend == "warp" else (0,))
    base = env.unwrapped
    model = base.mjm if backend == "warp" else base.model
    # Calibrate at two meters with a suitable near plane so OpenGL depth-buffer
    # quantization does not dominate the metric-distance assertion.
    model.vis.map.znear = 0.1 / model.stat.extent
    far = model.vis.map.zfar * model.stat.extent
    distance = 2 * far if view == "beyond_far" else 2.0
    if backend == "mujoco":
        cam = base._overhead_obs_cam
        cam.lookat[:] = [10, 0, 2 * distance if view == "sky" else 0]
        cam.distance = distance
        cam.elevation = 90 if view == "sky" else -90
    else:
        import torch
        import warp as wp

        camera_id = base._cam_specs[0][1]
        wp.to_torch(base.data.cam_xpos)[:, camera_id] = torch.tensor([10.0, 0, distance])
        rotation = np.diag([1.0, -1, -1]) if view == "sky" else np.eye(3)
        wp.to_torch(base.data.cam_xmat)[:, camera_id] = torch.tensor(rotation, dtype=torch.float32)
    obs = base._render_camera_images() if backend == "warp" else base._get_obs()
    depth = _numpy(obs["overhead_camera_depth"])
    expected = 2.0 if view == "floor" else far
    # A frontoparallel plane has constant axial depth, including off-center rays.
    np.testing.assert_allclose(depth, expected, rtol=1e-5, atol=1e-5)
