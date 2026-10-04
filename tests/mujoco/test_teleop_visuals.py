"""Runtime regression tests for teleop colors and recorded camera views."""

import mujoco
import numpy as np
import pytest

from so101_nexus.constants import COLOR_MAP
from so101_nexus.teleop.config_customization import TeleopConfigOverrides
from so101_nexus.teleop.session import _recording_env_kwargs, build_sim_follower_config


@pytest.mark.parametrize("colors", [("black",), ("red", "blue")])
def test_recording_robot_color_reaches_visual_materials(env_factory, colors):
    kwargs = _recording_env_kwargs(
        "MuJoCoTouch-v1",
        (32, 24),
        (48, 32),
        overrides=TeleopConfigOverrides(robot_colors=colors),
    )
    env = env_factory(**kwargs)
    model = env.unwrapped.model
    plastic = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "upper_arm_so101_v1_material")
    motor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "sts3215_03a_v1_material")
    motor_rgba = model.mat_rgba[motor].copy()
    sampled = []
    for seed in range(6):
        env.reset(seed=seed)
        color = model.mat_rgba[plastic].copy()
        assert any(np.allclose(color, COLOR_MAP[name]) for name in colors)
        np.testing.assert_array_equal(model.mat_rgba[motor], motor_rgba)
        env.reset(seed=seed)
        np.testing.assert_array_equal(model.mat_rgba[plastic], color)
        sampled.append(tuple(color))
    assert len(set(sampled)) == len(colors)


def test_recording_side_camera_produces_rgb_and_seeded_pose(env_factory, tmp_path):
    config = build_sim_follower_config(
        env_id="MuJoCoTouch-v1",
        robot_id="teleop_sim",
        wrist_wh=(32, 24),
        overhead_wh=(48, 32),
        calibration_dir=tmp_path,
    )
    assert "side" in config.cameras
    env = env_factory(render_mode="rgb_array", **config.env_kwargs)
    from so101_nexus.lerobot_adapter.sim_camera import SimCamera

    camera = SimCamera(config.cameras["side"])
    camera.bind_env(env)
    env.reset(seed=42)
    camera.connect()
    frame = camera.read()
    assert frame.shape == (32, 48, 3)
    assert frame.dtype == np.uint8
    assert env.unwrapped.config.render.camera == "side"
    first = env.unwrapped._render_camera_params().copy()
    env.reset(seed=42)
    assert env.unwrapped._render_camera_params()["azimuth"] == first["azimuth"]
    env.reset(seed=43)
    assert env.unwrapped._render_camera_params()["azimuth"] != first["azimuth"]
