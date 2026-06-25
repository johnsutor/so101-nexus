"""Cross-backend camera observation parity (structure, not pixels).

The Warp ray tracer and the MuJoCo OpenGL rasterizer produce different pixels, so
this asserts that observation keys, dtypes, and shapes match for the same config.
Skips when MuJoCo camera rendering (an OpenGL/EGL context) is unavailable.
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
