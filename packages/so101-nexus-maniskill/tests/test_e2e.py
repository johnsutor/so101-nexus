"""End-to-end tests for all ManiSkill environments."""

from __future__ import annotations

import gymnasium as gym
import pytest
import torch
from mani_skill import ASSET_DIR

import so101_nexus_maniskill  # noqa: F401
from so101_nexus_core.config import (
    LookAtConfig,
    MoveConfig,
    PickAndPlaceConfig,
    PickConfig,
    ReachConfig,
)
from so101_nexus_core.constants import CUBE_COLOR_MAP, YCB_OBJECTS
from so101_nexus_core.objects import CubeObject, YCBObject
from so101_nexus_core.observations import (
    EndEffectorPose,
    GazeDirection,
    JointPositions,
    OverheadCamera,
    TargetOffset,
    WristCamera,
)

N_STEPS = 3

BASE_KWARGS = {"obs_mode": "state", "num_envs": 1, "render_mode": None}

REACH_ENV_IDS = ["ManiSkillReachSO100-v1", "ManiSkillReachSO101-v1"]
LOOKAT_ENV_IDS = ["ManiSkillLookAtSO100-v1", "ManiSkillLookAtSO101-v1"]
MOVE_ENV_IDS = ["ManiSkillMoveSO100-v1", "ManiSkillMoveSO101-v1"]
PICK_LIFT_ENV_IDS = ["ManiSkillPickLiftSO100-v1", "ManiSkillPickLiftSO101-v1"]
PICK_AND_PLACE_ENV_IDS = [
    "ManiSkillPickAndPlaceSO100-v1",
    "ManiSkillPickAndPlaceSO101-v1",
]

ALL_ENV_IDS = (
    REACH_ENV_IDS + LOOKAT_ENV_IDS + MOVE_ENV_IDS + PICK_LIFT_ENV_IDS + PICK_AND_PLACE_ENV_IDS
)

CUBE_COLORS = list(CUBE_COLOR_MAP.keys())
YCB_MODEL_IDS = list(YCB_OBJECTS.keys())
MOVE_DIRECTIONS = ["up", "down", "left", "right", "forward", "backward"]


def _has_ycb_assets() -> bool:
    """Check whether ManiSkill YCB assets are available on disk."""
    manifest = ASSET_DIR / "assets" / "mani_skill2_ycb" / "info_pick_v0.json"
    return manifest.exists()


def _run_episode(env, n_steps: int = N_STEPS):
    """Reset env, take n_steps random actions, and return final (obs, info)."""
    obs, info = env.reset()
    for _ in range(n_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert reward is not None
    return obs, info


@pytest.mark.parametrize("env_id", ALL_ENV_IDS)
def test_default_config(env_id):
    """Each environment can be created, reset, and stepped with defaults."""
    env = gym.make(env_id, **BASE_KWARGS)
    _run_episode(env)
    assert isinstance(env.unwrapped.task_description, str)
    assert len(env.unwrapped.task_description) > 0
    env.close()


@pytest.mark.parametrize("env_id", ALL_ENV_IDS)
def test_action_space_shape(env_id):
    """All environments have a 6-DOF action space."""
    env = gym.make(env_id, **BASE_KWARGS)
    assert env.action_space.shape == (6,)
    env.close()


@pytest.mark.parametrize("env_id", ALL_ENV_IDS)
def test_observation_in_space(env_id):
    """Observations returned by reset are in the observation space."""
    env = gym.make(env_id, **BASE_KWARGS)
    obs, _ = env.reset()
    assert env.observation_space.contains(obs)
    env.close()


_reach_obs_params = [
    (JointPositions, "joint_positions"),
    (EndEffectorPose, "tcp_pose"),
    (TargetOffset, "target_offset"),
]


@pytest.mark.parametrize("env_id", REACH_ENV_IDS)
@pytest.mark.parametrize(
    "obs_cls,expected_key",
    _reach_obs_params,
    ids=[cls.__name__ for cls, _ in _reach_obs_params],
)
def test_reach_observation_component(env_id, obs_cls, expected_key):
    """Each valid observation component works on Reach envs."""
    config = ReachConfig(observations=[JointPositions(), obs_cls()])
    env = gym.make(env_id, config=config, **BASE_KWARGS)
    env.reset()
    _, _, _, _, info = env.step(env.action_space.sample())
    if obs_cls is not JointPositions:
        extra = env.unwrapped._get_obs_extra(info)
        assert expected_key in extra
    _run_episode(env)
    env.close()


_lookat_obs_params = [
    (JointPositions, "joint_positions"),
    (EndEffectorPose, "tcp_pose"),
    (GazeDirection, "gaze_direction"),
]


@pytest.mark.parametrize("env_id", LOOKAT_ENV_IDS)
@pytest.mark.parametrize(
    "obs_cls,expected_key",
    _lookat_obs_params,
    ids=[cls.__name__ for cls, _ in _lookat_obs_params],
)
def test_lookat_observation_component(env_id, obs_cls, expected_key):
    """Each valid observation component works on LookAt envs."""
    config = LookAtConfig(observations=[JointPositions(), obs_cls()])
    env = gym.make(env_id, config=config, **BASE_KWARGS)
    env.reset()
    _, _, _, _, info = env.step(env.action_space.sample())
    if obs_cls is not JointPositions:
        extra = env.unwrapped._get_obs_extra(info)
        assert expected_key in extra
    _run_episode(env)
    env.close()


_move_obs_params = [
    (JointPositions, "joint_positions"),
    (EndEffectorPose, "tcp_pose"),
    (TargetOffset, "target_offset"),
]


@pytest.mark.parametrize("env_id", MOVE_ENV_IDS)
@pytest.mark.parametrize(
    "obs_cls,expected_key",
    _move_obs_params,
    ids=[cls.__name__ for cls, _ in _move_obs_params],
)
def test_move_observation_component(env_id, obs_cls, expected_key):
    """Each valid observation component works on Move envs."""
    config = MoveConfig(observations=[JointPositions(), obs_cls()])
    env = gym.make(env_id, config=config, **BASE_KWARGS)
    env.reset()
    _, _, _, _, info = env.step(env.action_space.sample())
    if obs_cls is not JointPositions:
        extra = env.unwrapped._get_obs_extra(info)
        assert expected_key in extra
    _run_episode(env)
    env.close()


@pytest.mark.parametrize("env_id", PICK_LIFT_ENV_IDS)
@pytest.mark.parametrize("color", CUBE_COLORS)
def test_pick_cube_color(env_id, color):
    """PickLift works with every non-gray cube colour."""
    config = PickConfig(objects=[CubeObject(color=color)])  # type: ignore[arg-type]
    env = gym.make(env_id, config=config, **BASE_KWARGS)
    env.reset()
    assert color in env.unwrapped.task_description
    _run_episode(env)
    env.close()


@pytest.mark.parametrize("env_id", PICK_LIFT_ENV_IDS)
@pytest.mark.parametrize("model_id", YCB_MODEL_IDS)
def test_pick_ycb_object(env_id, model_id):
    """PickLift works with every YCB object model."""
    if not _has_ycb_assets():
        pytest.skip("ManiSkill YCB assets not available")
    config = PickConfig(objects=[YCBObject(model_id=model_id)])
    env = gym.make(env_id, config=config, **BASE_KWARGS)
    env.reset()
    expected_name = YCB_OBJECTS[model_id]
    assert expected_name in env.unwrapped.task_description
    _run_episode(env)
    env.close()


@pytest.mark.parametrize("env_id", PICK_LIFT_ENV_IDS)
def test_pick_multiple_cubes_with_distractors(env_id):
    """PickLift with a pool of cubes and distractors spawns correctly."""
    objects: list[CubeObject] = [
        CubeObject(color="red"),
        CubeObject(color="blue"),
        CubeObject(color="green"),
    ]
    config = PickConfig(objects=objects, n_distractors=2)  # type: ignore[arg-type]
    env = gym.make(env_id, config=config, **BASE_KWARGS)
    env.reset()
    _run_episode(env)
    env.close()


@pytest.mark.parametrize("env_id", LOOKAT_ENV_IDS)
@pytest.mark.parametrize("color", CUBE_COLORS)
def test_look_at_cube_color(env_id, color):
    """LookAt works with every non-gray cube colour."""
    config = LookAtConfig(objects=[CubeObject(color=color)])  # type: ignore[arg-type]
    env = gym.make(env_id, config=config, **BASE_KWARGS)
    env.reset()
    assert color in env.unwrapped.task_description
    _run_episode(env)
    env.close()


@pytest.mark.parametrize("env_id", MOVE_ENV_IDS)
@pytest.mark.parametrize("direction", MOVE_DIRECTIONS)
def test_move_direction(env_id, direction):
    """Move works with every cardinal direction."""
    config = MoveConfig(direction=direction)  # type: ignore[arg-type]
    env = gym.make(env_id, config=config, **BASE_KWARGS)
    env.reset()
    assert direction in env.unwrapped.task_description
    _run_episode(env)
    env.close()


@pytest.mark.parametrize("env_id", PICK_AND_PLACE_ENV_IDS)
@pytest.mark.parametrize("cube_color", CUBE_COLORS)
def test_pick_and_place_cube_colors(env_id, cube_color):
    """PickAndPlace works with every cube colour."""
    target_color = "blue" if cube_color != "blue" else "red"
    config = PickAndPlaceConfig(cube_colors=cube_color, target_colors=target_color)
    env = gym.make(env_id, config=config, **BASE_KWARGS)
    _run_episode(env)
    env.close()


@pytest.mark.parametrize("env_id", PICK_AND_PLACE_ENV_IDS)
@pytest.mark.parametrize("target_color", CUBE_COLORS)
def test_pick_and_place_target_colors(env_id, target_color):
    """PickAndPlace works with every target disc colour."""
    cube_color = "red" if target_color != "red" else "blue"
    config = PickAndPlaceConfig(cube_colors=cube_color, target_colors=target_color)
    env = gym.make(env_id, config=config, **BASE_KWARGS)
    _run_episode(env)
    env.close()


@pytest.mark.parametrize("env_id", ALL_ENV_IDS)
def test_multiple_resets(env_id):
    """Each environment can be reset multiple times without errors."""
    env = gym.make(env_id, **BASE_KWARGS)
    for _ in range(5):
        obs, _info = env.reset()
        assert obs is not None
        env.step(env.action_space.sample())
    env.close()


@pytest.mark.parametrize("env_id", ALL_ENV_IDS)
def test_reward_bounds(env_id):
    """Reward stays within the expected [0, 1] range over multiple steps."""
    env = gym.make(env_id, **BASE_KWARGS)
    env.reset()
    for _ in range(10):
        _, reward, terminated, _, _ = env.step(env.action_space.sample())
        assert (reward >= 0).all(), f"Reward below 0 for {env_id}"
        assert (reward <= 1).all(), f"Reward above 1 for {env_id}"
        if terminated.any() if isinstance(terminated, torch.Tensor) else terminated:
            env.reset()
    env.close()


# ---------------------------------------------------------------------------
# Observation-driven camera tests
# ---------------------------------------------------------------------------

_obs_cam_env_params = [
    (REACH_ENV_IDS[0], ReachConfig),
    (LOOKAT_ENV_IDS[0], LookAtConfig),
    (MOVE_ENV_IDS[0], MoveConfig),
    (PICK_LIFT_ENV_IDS[0], PickConfig),
    (PICK_AND_PLACE_ENV_IDS[0], PickAndPlaceConfig),
]


class TestObservationDrivenCameras:
    """Test camera observation components create correct sensor configs."""

    @pytest.mark.parametrize(
        "env_id,config_cls",
        _obs_cam_env_params,
        ids=[eid for eid, _ in _obs_cam_env_params],
    )
    def test_overhead_camera_sensor(self, env_id, config_cls):
        """OverheadCamera observation component creates an overhead_camera sensor."""
        config = config_cls(observations=[JointPositions(), OverheadCamera(width=64, height=48)])
        env = gym.make(env_id, config=config, **BASE_KWARGS)
        sensor_names = [cfg.uid for cfg in env.unwrapped._default_sensor_configs]
        assert "overhead_camera" in sensor_names
        _run_episode(env)
        env.close()

    @pytest.mark.parametrize(
        "env_id,config_cls",
        _obs_cam_env_params,
        ids=[eid for eid, _ in _obs_cam_env_params],
    )
    def test_wrist_camera_sensor(self, env_id, config_cls):
        """WristCamera observation component creates a wrist_camera sensor."""
        config = config_cls(observations=[JointPositions(), WristCamera(width=64, height=48)])
        env = gym.make(env_id, config=config, **BASE_KWARGS)
        sensor_names = [cfg.uid for cfg in env.unwrapped._default_sensor_configs]
        assert "wrist_camera" in sensor_names
        _run_episode(env)
        env.close()

    @pytest.mark.parametrize(
        "env_id,config_cls",
        _obs_cam_env_params,
        ids=[eid for eid, _ in _obs_cam_env_params],
    )
    def test_both_cameras_sensor(self, env_id, config_cls):
        """Both camera components create both sensors."""
        config = config_cls(
            observations=[
                JointPositions(),
                WristCamera(width=64, height=48),
                OverheadCamera(width=32, height=24),
            ]
        )
        env = gym.make(env_id, config=config, **BASE_KWARGS)
        sensor_names = [cfg.uid for cfg in env.unwrapped._default_sensor_configs]
        assert "wrist_camera" in sensor_names
        assert "overhead_camera" in sensor_names
        _run_episode(env)
        env.close()

    @pytest.mark.parametrize(
        "env_id,config_cls",
        _obs_cam_env_params,
        ids=[eid for eid, _ in _obs_cam_env_params],
    )
    def test_no_camera_no_obs_sensors(self, env_id, config_cls):
        """No camera components means no overhead/wrist observation sensors."""
        config = config_cls(observations=[JointPositions()])
        env = gym.make(env_id, config=config, **BASE_KWARGS)
        sensor_names = [cfg.uid for cfg in env.unwrapped._default_sensor_configs]
        assert "overhead_camera" not in sensor_names
        assert "wrist_camera" not in sensor_names
        _run_episode(env)
        env.close()
