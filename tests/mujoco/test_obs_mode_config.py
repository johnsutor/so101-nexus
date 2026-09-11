"""Tests for obs_mode config validation."""

import numpy as np
import pytest

from so101_nexus import (
    EnvironmentConfig,
    LookAtConfig,
    MoveConfig,
    PickAndPlaceConfig,
    PickConfig,
    TouchConfig,
)
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

_PICK_STATE_OBS = [
    JointPositions,
    JointVelocities,
    EndEffectorPose,
    GraspState,
    ObjectPose,
    ObjectOffset,
]
_PICK_AND_PLACE_STATE_OBS = [
    JointPositions,
    JointVelocities,
    EndEffectorPose,
    GraspState,
    TargetPosition,
    ObjectPose,
    ObjectOffset,
    TargetOffset,
]
_TOUCH_STATE_OBS = [JointPositions, JointVelocities, EndEffectorPose, ObjectOffset]

_PICK_STATE_SIZE = 30
_PICK_AND_PLACE_STATE_SIZE = 36
_TOUCH_STATE_SIZE = 22


class TestObsModeConfig:
    def test_default_obs_mode_is_state(self):
        cfg = EnvironmentConfig()
        assert cfg.obs_mode == "state"

    def test_invalid_obs_mode_rejected(self):
        with pytest.raises(ValueError, match="obs_mode"):
            EnvironmentConfig(obs_mode="invalid")


@pytest.mark.parametrize(
    "task,config_cls,components,state_size,obs_mode",
    [
        pytest.param(
            "PickLift", PickConfig, _PICK_STATE_OBS, _PICK_STATE_SIZE, "visual", id="pick-visual"
        ),
        pytest.param(
            "PickLift", PickConfig, _PICK_STATE_OBS, _PICK_STATE_SIZE, "state", id="pick-state"
        ),
        pytest.param(
            "PickAndPlace",
            PickAndPlaceConfig,
            _PICK_AND_PLACE_STATE_OBS,
            _PICK_AND_PLACE_STATE_SIZE,
            "visual",
            id="place-visual",
        ),
        pytest.param(
            "PickAndPlace",
            PickAndPlaceConfig,
            _PICK_AND_PLACE_STATE_OBS,
            _PICK_AND_PLACE_STATE_SIZE,
            "state",
            id="place-state",
        ),
        pytest.param(
            "Touch", TouchConfig, _TOUCH_STATE_OBS, _TOUCH_STATE_SIZE, "visual", id="touch-visual"
        ),
        pytest.param("LookAt", LookAtConfig, [JointPositions], 6, "visual", id="lookat-visual"),
        pytest.param("Move", MoveConfig, [JointPositions], 6, "visual", id="move-visual"),
    ],
)
def test_obs_mode_routes_state_and_privileged_state(
    task, config_cls, components, state_size, obs_mode, env_factory
):
    config = config_cls(
        obs_mode=obs_mode,
        observations=[cls() for cls in components] + [WristCamera(width=64, height=48)],
    )
    env = env_factory(task=task, config=config)
    obs, info = env.reset(seed=0)
    assert isinstance(obs, dict)
    assert obs["state"].dtype == np.float32
    if obs_mode == "visual":
        assert obs["state"].shape == (6,)
        assert info["privileged_state"].shape == (state_size,)
    else:
        assert obs["state"].shape == (state_size,)
        assert "privileged_state" not in info
