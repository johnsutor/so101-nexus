"""Cross-backend camera observation parity.

Structure (keys, dtypes, shapes) matches exactly. Pixels do not: the Warp ray
tracer and the MuJoCo OpenGL rasterizer shade the same scene differently, and
``so101_nexus.testing.assert_render_parity`` exists so consumers can measure that
gap instead of discovering it from a failed training run. Skips when MuJoCo
camera rendering (an OpenGL/EGL context) is unavailable.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.warp

WRIST_W, WRIST_H = 24, 16
OVER_W, OVER_H = 20, 12
NUM_ENVS = 2


def _camera_observations(obs_mode):
    from so101_nexus.observations import (
        EndEffectorPose,
        GraspState,
        JointPositions,
        OverheadCamera,
        WristCamera,
    )

    return {
        "obs_mode": obs_mode,
        "observations": [
            JointPositions(),
            EndEffectorPose(),
            GraspState(),
            WristCamera(width=WRIST_W, height=WRIST_H),
            OverheadCamera(width=OVER_W, height=OVER_H),
        ],
    }


def _mujoco_obs(obs_mode):
    import gymnasium

    import so101_nexus.mujoco  # noqa: F401
    from so101_nexus.config import PickConfig

    try:
        env = gymnasium.make(
            "MuJoCoPickLift-v1", config=PickConfig(**_camera_observations(obs_mode))
        )
        obs, info = env.reset(seed=0)
    except Exception as exc:  # no GL/EGL context in this environment
        pytest.skip(f"MuJoCo camera rendering unavailable: {exc}")
    env.close()
    return obs, info


def _warp_obs(obs_mode):
    import so101_nexus.warp  # noqa: F401
    from so101_nexus.config import PickConfig
    from so101_nexus.warp.pick_env import WarpPickLiftVectorEnv

    env = WarpPickLiftVectorEnv(
        num_envs=NUM_ENVS,
        config=PickConfig(**_camera_observations(obs_mode)),
        device="cpu",
        seed=0,
    )
    obs, info = env.reset(seed=0)
    return obs, info


@pytest.mark.parametrize("obs_mode", ["state", "visual"])
def test_camera_obs_structure_matches_mujoco(obs_mode):
    mj_obs, mj_info = _mujoco_obs(obs_mode)
    wp_obs, wp_info = _warp_obs(obs_mode)

    assert set(wp_obs) == set(mj_obs)

    for key in ("wrist_camera", "overhead_camera"):
        mj_img, wp_img = mj_obs[key], wp_obs[key]
        assert mj_img.dtype == np.uint8
        assert str(wp_img.dtype) == "torch.uint8"
        # MuJoCo image is (H, W, 3); Warp batches to (N, H, W, 3).
        assert wp_img.shape == (NUM_ENVS, *mj_img.shape)

    mj_state, wp_state = mj_obs["state"], wp_obs["state"]
    assert mj_state.dtype == np.float32
    assert str(wp_state.dtype) == "torch.float32"
    assert wp_state.shape == (NUM_ENVS, *mj_state.shape)

    if obs_mode == "visual":
        assert "privileged_state" in mj_info
        assert "privileged_state" in wp_info
        assert wp_info["privileged_state"].shape == (NUM_ENVS, *mj_info["privileged_state"].shape)


def _parity_config_factory():
    from so101_nexus.config import PickAndPlaceConfig
    from so101_nexus.observations import (
        EndEffectorPose,
        GraspState,
        JointPositions,
        JointVelocities,
        ObjectOffset,
        ObjectPose,
        TargetOffset,
        TargetPosition,
        WristCamera,
    )

    def factory():
        return PickAndPlaceConfig(
            obs_mode="visual",
            observations=[
                JointPositions(),
                JointVelocities(),
                EndEffectorPose(),
                GraspState(),
                TargetPosition(),
                ObjectPose(),
                ObjectOffset(),
                TargetOffset(),
                # Domain randomization collapsed to single points so both backends
                # place the identical camera; otherwise this measures DR sampling.
                WristCamera(
                    width=64,
                    height=48,
                    fov_deg_range=(75.0, 75.0),
                    pitch_deg_range=(-17.0, -17.0),
                    pos_x_noise=0.0,
                    pos_y_noise=0.0,
                    pos_z_noise=0.0,
                ),
            ],
        )

    return factory


def _measure_parity():
    from so101_nexus.testing import measure_render_parity

    try:
        return measure_render_parity("PickAndPlace", _parity_config_factory())
    except Exception as exc:  # no GL/EGL context in this environment
        pytest.skip(f"MuJoCo camera rendering unavailable: {exc}")


def test_render_parity_measures_the_backends_at_identical_state():
    """The measurement is only meaningful if state is eliminated as a variable."""
    report = _measure_parity()

    assert report.max_qpos_diff == 0.0
    assert [c.name for c in report.cameras] == ["wrist_camera"]


def test_backends_share_a_background_colour():
    """mujoco_warp clears to a blue-tinted default that matches nothing in the
    model; the Warp backend overrides it to MuJoCo's black so the out-of-scene
    region, at least, is interchangeable."""
    import gymnasium

    import so101_nexus.warp  # noqa: F401

    factory = _parity_config_factory()
    envs = gymnasium.make_vec("WarpPickAndPlace-v1", num_envs=1, config=factory())
    try:
        obs, _ = envs.reset(seed=0)
        # Top-right corner of the wrist view looks past the table edge at nothing.
        corner = obs["wrist_camera"][0, :4, -4:].cpu().numpy()
    finally:
        envs.close()

    assert corner.max() == 0, f"expected a black background, got {corner.reshape(-1, 3).mean(0)}"


def test_render_parity_assertion_reports_the_measured_gap():
    """A tolerance tighter than the real gap must fail, and say by how much:
    the helper is worthless if the shipped divergence can pass it silently."""
    from so101_nexus.testing import assert_render_parity

    report = _measure_parity()
    gap = report.cameras[0].mean_abs_diff
    assert gap > 0.0, "backends rendered identical pixels; this test is vacuous"

    assert_render_parity("PickAndPlace", _parity_config_factory(), max_mean_abs_diff=gap + 1.0)
    with pytest.raises(AssertionError, match="mean abs pixel difference"):
        assert_render_parity("PickAndPlace", _parity_config_factory(), max_mean_abs_diff=gap / 2.0)
