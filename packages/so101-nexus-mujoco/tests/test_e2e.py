"""End-to-end tests for all MuJoCo environments."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

import so101_nexus_mujoco  # noqa: F401
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
    GraspState,
    JointPositions,
    ObjectOffset,
    ObjectPose,
    OverheadCamera,
    TargetOffset,
    TargetPosition,
    WristCamera,
)

N_STEPS = 3

REACH_OBS = [JointPositions, EndEffectorPose, TargetOffset]
LOOKAT_OBS = [JointPositions, EndEffectorPose, GazeDirection]
MOVE_OBS = [JointPositions, EndEffectorPose, TargetOffset]
PICK_OBS = [JointPositions, EndEffectorPose, GraspState, ObjectPose, ObjectOffset]
PICK_AND_PLACE_OBS = [
    JointPositions,
    EndEffectorPose,
    GraspState,
    TargetPosition,
    ObjectPose,
    ObjectOffset,
    TargetOffset,
]

OBS_SIZES: dict[type, int] = {
    JointPositions: 6,
    EndEffectorPose: 7,
    TargetOffset: 3,
    GazeDirection: 3,
    GraspState: 1,
    ObjectPose: 7,
    ObjectOffset: 3,
    TargetPosition: 3,
}

MUJOCO_ENV_IDS = [
    "MuJoCoReach-v1",
    "MuJoCoLookAt-v1",
    "MuJoCoMove-v1",
    "MuJoCoPickLift-v1",
    "MuJoCoPickAndPlace-v1",
]

CUBE_COLORS = list(CUBE_COLOR_MAP.keys())
YCB_MODEL_IDS = list(YCB_OBJECTS.keys())
MOVE_DIRECTIONS = ["up", "down", "left", "right", "forward", "backward"]
CONTROL_MODES = ["pd_joint_pos", "pd_joint_delta_pos", "pd_joint_target_delta_pos"]


def _run_episode(env, n_steps: int = N_STEPS):
    """Reset env, take n_steps random actions, and return final (obs, info)."""
    obs, info = env.reset()
    for _ in range(n_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(reward, (float, int, np.floating))
        assert isinstance(terminated, (bool, np.bool_))
        assert isinstance(truncated, (bool, np.bool_))
    return obs, info


@pytest.mark.parametrize("env_id", MUJOCO_ENV_IDS)
def test_default_config(env_id):
    """Each environment can be created, reset, and stepped with defaults."""
    env = gym.make(env_id)
    _obs, info = _run_episode(env)
    assert "success" in info
    assert isinstance(env.unwrapped.task_description, str)  # type: ignore[attr-defined]
    assert len(env.unwrapped.task_description) > 0  # type: ignore[attr-defined]
    env.close()


_single_obs_params = []
for _cls in REACH_OBS:
    _single_obs_params.append(("MuJoCoReach-v1", _cls, ReachConfig))
for _cls in LOOKAT_OBS:
    _single_obs_params.append(("MuJoCoLookAt-v1", _cls, LookAtConfig))
for _cls in MOVE_OBS:
    _single_obs_params.append(("MuJoCoMove-v1", _cls, MoveConfig))
for _cls in PICK_OBS:
    _single_obs_params.append(("MuJoCoPickLift-v1", _cls, PickConfig))
for _cls in PICK_AND_PLACE_OBS:
    _single_obs_params.append(("MuJoCoPickAndPlace-v1", _cls, PickAndPlaceConfig))


@pytest.mark.parametrize(
    "env_id,obs_cls,config_cls",
    _single_obs_params,
    ids=[f"{eid}-{ocls.__name__}" for eid, ocls, _ in _single_obs_params],
)
def test_single_observation_component(env_id, obs_cls, config_cls):
    """Each valid observation component works individually on each env."""
    config = config_cls(observations=[obs_cls()])
    env = gym.make(env_id, config=config)
    obs, _ = env.reset()
    expected = OBS_SIZES[obs_cls]
    assert obs.shape == (expected,), f"Expected ({expected},), got {obs.shape}"
    _run_episode(env)
    env.close()


_all_obs_params = [
    ("MuJoCoReach-v1", REACH_OBS, ReachConfig),
    ("MuJoCoLookAt-v1", LOOKAT_OBS, LookAtConfig),
    ("MuJoCoMove-v1", MOVE_OBS, MoveConfig),
    ("MuJoCoPickLift-v1", PICK_OBS, PickConfig),
    ("MuJoCoPickAndPlace-v1", PICK_AND_PLACE_OBS, PickAndPlaceConfig),
]


@pytest.mark.parametrize(
    "env_id,obs_classes,config_cls",
    _all_obs_params,
    ids=[eid for eid, _, _ in _all_obs_params],
)
def test_all_observation_components_combined(env_id, obs_classes, config_cls):
    """All valid observation components together produce the correct obs size."""
    config = config_cls(observations=[cls() for cls in obs_classes])
    env = gym.make(env_id, config=config)
    obs, _ = env.reset()
    expected = sum(OBS_SIZES[cls] for cls in obs_classes)
    assert obs.shape == (expected,)
    _run_episode(env)
    env.close()


@pytest.mark.parametrize("color", CUBE_COLORS)
def test_pick_cube_color(color):
    """PickLift works with every non-gray cube colour."""
    config = PickConfig(objects=[CubeObject(color=color)])  # type: ignore[arg-type]
    env = gym.make("MuJoCoPickLift-v1", config=config)
    env.reset()
    assert color in env.unwrapped.task_description  # type: ignore[attr-defined]
    _run_episode(env)
    env.close()


@pytest.mark.parametrize("model_id", YCB_MODEL_IDS)
def test_pick_ycb_object(model_id):
    """PickLift works with every YCB object model."""
    config = PickConfig(objects=[YCBObject(model_id=model_id)])
    env = gym.make("MuJoCoPickLift-v1", config=config)
    env.reset()
    expected_name = YCB_OBJECTS[model_id]
    assert expected_name in env.unwrapped.task_description  # type: ignore[attr-defined]
    _run_episode(env)
    env.close()


def test_pick_multiple_cubes_with_distractors():
    """PickLift with a pool of cubes and distractors spawns correctly."""
    objects: list[CubeObject] = [
        CubeObject(color="red"),
        CubeObject(color="blue"),
        CubeObject(color="green"),
    ]
    config = PickConfig(objects=objects, n_distractors=2)  # type: ignore[arg-type]
    env = gym.make("MuJoCoPickLift-v1", config=config)
    obs, _ = env.reset()
    assert obs.shape == (18,)
    _run_episode(env)
    env.close()


def test_pick_mixed_ycb_cubes_with_distractors():
    """PickLift with a mixed pool of YCB and cube objects."""
    objects = [
        YCBObject(model_id="011_banana"),
        CubeObject(color="blue"),
        YCBObject(model_id="058_golf_ball"),
    ]
    config = PickConfig(objects=objects, n_distractors=2)
    env = gym.make("MuJoCoPickLift-v1", config=config)
    _run_episode(env)
    env.close()


@pytest.mark.parametrize("color", CUBE_COLORS)
def test_look_at_cube_color(color):
    """LookAt works with every non-gray cube colour."""
    config = LookAtConfig(objects=[CubeObject(color=color)])  # type: ignore[arg-type]
    env = gym.make("MuJoCoLookAt-v1", config=config)
    env.reset()
    assert color in env.unwrapped.task_description  # type: ignore[attr-defined]
    _run_episode(env)
    env.close()


@pytest.mark.parametrize("direction", MOVE_DIRECTIONS)
def test_move_direction(direction):
    """Move works with every cardinal direction."""
    config = MoveConfig(direction=direction)  # type: ignore[arg-type]
    env = gym.make("MuJoCoMove-v1", config=config)
    env.reset()
    assert direction in env.unwrapped.task_description  # type: ignore[attr-defined]
    _run_episode(env)
    env.close()


@pytest.mark.parametrize("env_id", MUJOCO_ENV_IDS)
@pytest.mark.parametrize("control_mode", CONTROL_MODES)
def test_control_mode(env_id, control_mode):
    """Every environment works with every supported control mode."""
    env = gym.make(env_id, control_mode=control_mode)
    _run_episode(env)
    env.close()


@pytest.mark.parametrize("env_id", MUJOCO_ENV_IDS)
def test_multiple_resets(env_id):
    """Each environment can be reset multiple times without errors."""
    env = gym.make(env_id)
    for _ in range(5):
        obs, _info = env.reset()
        assert obs is not None
        env.step(env.action_space.sample())
    env.close()


@pytest.mark.parametrize("env_id", MUJOCO_ENV_IDS)
def test_reward_bounds(env_id):
    """Reward stays within the expected [-0.1, 1.0] range over multiple steps."""
    env = gym.make(env_id)
    env.reset()
    for _ in range(10):
        _, reward, terminated, _, _ = env.step(env.action_space.sample())
        assert -0.1 <= float(reward) <= 1.0, f"Reward {reward} out of bounds for {env_id}"
        if terminated:
            env.reset()
    env.close()


@pytest.mark.parametrize("cube_color", CUBE_COLORS)
def test_pick_and_place_cube_colors(cube_color):
    """PickAndPlace works with every cube colour."""
    target_color = "blue" if cube_color != "blue" else "red"
    config = PickAndPlaceConfig(cube_colors=cube_color, target_colors=target_color)
    env = gym.make("MuJoCoPickAndPlace-v1", config=config)
    _run_episode(env)
    env.close()


@pytest.mark.parametrize("target_color", CUBE_COLORS)
def test_pick_and_place_target_colors(target_color):
    """PickAndPlace works with every target disc colour."""
    cube_color = "red" if target_color != "red" else "blue"
    config = PickAndPlaceConfig(cube_colors=cube_color, target_colors=target_color)
    env = gym.make("MuJoCoPickAndPlace-v1", config=config)
    _run_episode(env)
    env.close()


@pytest.mark.parametrize("env_id", MUJOCO_ENV_IDS)
def test_seeded_reset_reproducible(env_id):
    """Resetting with the same seed produces identical observations."""
    env = gym.make(env_id)
    obs1, _ = env.reset(seed=42)
    obs2, _ = env.reset(seed=42)
    np.testing.assert_array_equal(obs1, obs2)
    env.close()


# ---------------------------------------------------------------------------
# Observation-driven camera tests
# ---------------------------------------------------------------------------

_ENV_CLS_MAP: dict[str, tuple] = {}  # populated lazily


def _get_env_cls_map():
    """Lazily import env classes to avoid top-level import overhead."""
    if not _ENV_CLS_MAP:
        from so101_nexus_mujoco.look_at_env import LookAtEnv
        from so101_nexus_mujoco.move_env import MoveEnv
        from so101_nexus_mujoco.pick_and_place import PickAndPlaceEnv
        from so101_nexus_mujoco.pick_env import PickLiftEnv
        from so101_nexus_mujoco.reach_env import ReachEnv

        _ENV_CLS_MAP.update(
            {
                "reach": (ReachEnv, ReachConfig),
                "lookat": (LookAtEnv, LookAtConfig),
                "move": (MoveEnv, MoveConfig),
                "pick": (PickLiftEnv, PickConfig),
                "pickandplace": (PickAndPlaceEnv, PickAndPlaceConfig),
            }
        )
    return _ENV_CLS_MAP


_OBS_CAM_PARAMS = ["reach", "lookat", "move", "pick", "pickandplace"]


class TestObservationDrivenCameras:
    """E2E tests for OverheadCamera and WristCamera observation components."""

    @pytest.mark.parametrize("env_key", _OBS_CAM_PARAMS)
    def test_overhead_camera_obs(self, env_key):
        env_cls, config_cls = _get_env_cls_map()[env_key]
        cfg = config_cls(observations=[JointPositions(), OverheadCamera(width=64, height=48)])
        env = env_cls(config=cfg)
        obs, info = env.reset()
        assert isinstance(obs, dict)
        assert "state" in obs and "overhead_camera" in obs
        assert obs["overhead_camera"].shape == (48, 64, 3)
        assert obs["overhead_camera"].dtype == np.uint8
        obs2, _, _, _, _ = env.step(env.action_space.sample())
        assert "overhead_camera" in obs2
        env.close()

    @pytest.mark.parametrize("env_key", _OBS_CAM_PARAMS)
    def test_wrist_camera_obs_component(self, env_key):
        env_cls, config_cls = _get_env_cls_map()[env_key]
        cfg = config_cls(observations=[JointPositions(), WristCamera(width=64, height=48)])
        env = env_cls(config=cfg)
        obs, info = env.reset()
        assert isinstance(obs, dict)
        assert "state" in obs and "wrist_camera" in obs
        assert obs["wrist_camera"].shape == (48, 64, 3)
        assert obs["wrist_camera"].dtype == np.uint8
        env.close()

    @pytest.mark.parametrize("env_key", _OBS_CAM_PARAMS)
    def test_both_cameras_obs(self, env_key):
        env_cls, config_cls = _get_env_cls_map()[env_key]
        cfg = config_cls(
            observations=[
                JointPositions(),
                WristCamera(width=64, height=48),
                OverheadCamera(width=32, height=24),
            ]
        )
        env = env_cls(config=cfg)
        obs, info = env.reset()
        assert isinstance(obs, dict)
        assert set(obs.keys()) == {"state", "wrist_camera", "overhead_camera"}
        assert obs["wrist_camera"].shape == (48, 64, 3)
        assert obs["overhead_camera"].shape == (24, 32, 3)
        env.close()

    @pytest.mark.parametrize("env_key", _OBS_CAM_PARAMS)
    def test_no_camera_flat_obs(self, env_key):
        env_cls, config_cls = _get_env_cls_map()[env_key]
        cfg = config_cls(observations=[JointPositions()])
        env = env_cls(config=cfg)
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (6,)
        env.close()

    @pytest.mark.parametrize("env_key", _OBS_CAM_PARAMS)
    def test_visual_with_overhead(self, env_key):
        env_cls, config_cls = _get_env_cls_map()[env_key]
        cfg = config_cls(
            obs_mode="visual",
            observations=[JointPositions(), OverheadCamera(width=64, height=48)],
        )
        env = env_cls(config=cfg)
        obs, info = env.reset()
        assert isinstance(obs, dict)
        assert obs["state"].shape == (6,)
        assert "privileged_state" in info
        assert "overhead_camera" in obs
        env.close()

    @pytest.mark.parametrize("env_key", ["reach", "pick"])
    def test_visual_with_both(self, env_key):
        env_cls, config_cls = _get_env_cls_map()[env_key]
        cfg = config_cls(
            obs_mode="visual",
            observations=[
                JointPositions(),
                WristCamera(width=64, height=48),
                OverheadCamera(width=32, height=24),
            ],
        )
        env = env_cls(config=cfg)
        obs, info = env.reset()
        assert obs["state"].shape == (6,)
        assert "privileged_state" in info
        assert "wrist_camera" in obs and "overhead_camera" in obs
        env.close()

    def test_render_independent_of_overhead_obs(self):
        from so101_nexus_mujoco.reach_env import ReachEnv

        cfg = ReachConfig(observations=[JointPositions(), OverheadCamera(width=64, height=48)])
        env = ReachEnv(config=cfg, render_mode="rgb_array")
        env.reset()
        frame = env.render()
        assert frame is not None
        assert frame.dtype == np.uint8
        env.close()
