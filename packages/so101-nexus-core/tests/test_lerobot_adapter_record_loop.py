"""End-to-end LeRobot record-loop smoke test for the simulator follower."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from so101_nexus_core.config import SO101_JOINT_NAMES

if TYPE_CHECKING:
    from pathlib import Path

gym = pytest.importorskip("gymnasium")
teleoperator_config_module = pytest.importorskip("lerobot.teleoperators.config")
teleoperator_module = pytest.importorskip("lerobot.teleoperators.teleoperator")
TeleoperatorConfig = teleoperator_config_module.TeleoperatorConfig
Teleoperator = teleoperator_module.Teleoperator


class _RecordLoopEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        render_mode: str | None = None,
        control_mode: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.control_mode = control_mode
        self.qpos = np.zeros(6, dtype=np.float64)
        self._ctrl_low = np.array([-2.0, -2.0, -2.0, -2.0, -2.0, -0.2], dtype=np.float64)
        self._ctrl_high = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 1.2], dtype=np.float64)
        self.action_space = gym.spaces.Box(
            low=self._ctrl_low.astype(np.float32),
            high=self._ctrl_high.astype(np.float32),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(6,),
            dtype=np.float64,
        )

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.qpos = np.zeros(6, dtype=np.float64)
        return self.qpos.copy(), {}

    def step(self, action):
        self.qpos = np.asarray(action, dtype=np.float64)
        return self.qpos.copy(), 0.0, False, False, {}

    def _get_current_qpos(self) -> np.ndarray:
        return self.qpos.copy()


@TeleoperatorConfig.register_subclass("fake_record_loop")
@dataclass(kw_only=True)
class _FakeTeleopConfig(TeleoperatorConfig):
    pass


class _FakeTeleop(Teleoperator):
    config_class = _FakeTeleopConfig
    name = "fake_record_loop"

    def __init__(self, action: dict[str, float]) -> None:
        super().__init__(_FakeTeleopConfig(id="fake_record_loop"))
        self._action = action

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{name}.pos": float for name in SO101_JOINT_NAMES}

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return True

    @property
    def is_calibrated(self) -> bool:
        return True

    def connect(self, calibrate: bool = True) -> None:
        pass

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def get_action(self) -> dict[str, float]:
        return dict(self._action)

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    def disconnect(self) -> None:
        pass


@pytest.fixture
def fake_env_id() -> str:
    env_id = "SO101NexusRecordLoopFake-v0"
    gym.register(id=env_id, entry_point=_RecordLoopEnv)
    try:
        yield env_id
    finally:
        gym.envs.registration.registry.pop(env_id, None)


def _dataset_features(robot):
    from lerobot.datasets.pipeline_features import (
        aggregate_pipeline_dataset_features,
        create_initial_features,
    )
    from lerobot.datasets.utils import combine_feature_dicts
    from lerobot.processor import make_default_processors

    teleop_action_processor, _, robot_observation_processor = make_default_processors()
    return combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=False,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=False,
        ),
    )


def test_lerobot_record_loop_writes_sim_follower_dataset(
    tmp_path: Path,
    fake_env_id: str,
) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.processor import make_default_processors
    from lerobot.scripts.lerobot_record import record_loop

    from so101_nexus_core.lerobot_adapter import SimSOFollower, SimSOFollowerConfig
    from so101_nexus_core.lerobot_adapter.synthetic_calibration import (
        write_synthetic_calibration,
    )

    calibration_dir = tmp_path / "calibration"
    write_synthetic_calibration(calibration_dir, "sim_test")
    robot = SimSOFollower(
        SimSOFollowerConfig(
            id="sim_test",
            calibration_dir=calibration_dir,
            env_id=fake_env_id,
        )
    )
    action = {f"{name}.pos": 0.0 for name in SO101_JOINT_NAMES}
    teleop = _FakeTeleop(action)
    dataset_root = tmp_path / "datasets"
    repo_id = "test/sim-record-loop"
    dataset = None

    try:
        robot.connect()
        features = _dataset_features(robot)
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=5,
            features=features,
            root=dataset_root,
            robot_type=robot.name,
            use_videos=False,
        )
        teleop_action_processor, robot_action_processor, robot_observation_processor = (
            make_default_processors()
        )

        record_loop(
            robot=robot,
            events={"exit_early": False, "rerecord_episode": False},
            fps=5,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            dataset=dataset,
            teleop=teleop,
            control_time_s=0.25,
            single_task="hold neutral pose",
        )
        dataset.save_episode()
        dataset.finalize()
        dataset = None

        reloaded = LeRobotDataset(repo_id, root=dataset_root)

        assert "shoulder_pan.pos" in reloaded.features["observation.state"]["names"]
        assert "shoulder_pan.pos" in reloaded.features["action"]["names"]
        assert "task_index" in reloaded.features
        assert reloaded.num_frames >= 1
        assert list(reloaded.meta.tasks.index) == ["hold neutral pose"]
        assert int(reloaded.meta.tasks.loc["hold neutral pose", "task_index"]) == 0
    finally:
        if dataset is not None:
            dataset.finalize()
        if robot.is_connected:
            robot.disconnect()
