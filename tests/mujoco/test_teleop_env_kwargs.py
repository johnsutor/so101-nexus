"""Backend-dependent tests for teleop env-kwarg resolution.

These live in the mujoco package because they exercise
``_recording_env_kwargs`` against real registered MuJoCo envs.
"""

from __future__ import annotations

import so101_nexus.mujoco  # noqa: F401 - registers gym envs
from so101_nexus.objects import CubeObject, YCBObject
from so101_nexus.observations import OverheadCamera, WristCamera
from so101_nexus.teleop.config_customization import TeleopConfigOverrides
from so101_nexus.teleop.session import RECORDING_CONTROL_MODE, _recording_env_kwargs


def test_recording_env_kwargs_overrides_wrist_camera_size() -> None:
    kwargs = _recording_env_kwargs("MuJoCoPickLift-v1", (480, 512), (640, 480))
    observations = kwargs["config"].observations

    wrist = [o for o in observations if isinstance(o, WristCamera)]
    assert len(wrist) == 1
    assert wrist[0].width == 480
    assert wrist[0].height == 512


def test_recording_env_kwargs_preserves_registered_env_kwargs() -> None:
    kwargs = _recording_env_kwargs("MuJoCoPickAndPlace-v1", (640, 360), (640, 480))
    observations = kwargs["config"].observations

    wrist = [o for o in observations if isinstance(o, WristCamera)]
    assert len(wrist) == 1
    assert wrist[0].width == 640
    assert wrist[0].height == 360


def test_recording_env_kwargs_wires_both_cameras() -> None:
    kwargs = _recording_env_kwargs("MuJoCoPickLift-v1", (320, 240), (800, 600))
    observations = kwargs["config"].observations

    wrist = [o for o in observations if isinstance(o, WristCamera)]
    overhead = [o for o in observations if isinstance(o, OverheadCamera)]

    assert len(wrist) == 1
    assert wrist[0].width == 320
    assert wrist[0].height == 240

    assert len(overhead) == 1
    assert overhead[0].width == 800
    assert overhead[0].height == 600


def test_recording_env_kwargs_applies_pick_overrides() -> None:
    kwargs = _recording_env_kwargs(
        "MuJoCoPickLift-v1",
        (320, 240),
        (640, 480),
        overrides=TeleopConfigOverrides(
            object_specs=("cube:green", "ycb:009_gelatin_box"),
            n_distractors=1,
        ),
    )

    assert kwargs["config"].n_distractors == 1
    assert isinstance(kwargs["config"].objects[0], CubeObject)
    assert kwargs["config"].objects[0].color == "green"
    assert isinstance(kwargs["config"].objects[1], YCBObject)
    assert kwargs["config"].objects[1].model_id == "009_gelatin_box"


def test_recording_env_kwargs_applies_stack_cube_overrides() -> None:
    kwargs = _recording_env_kwargs(
        "MuJoCoStackCube-v1",
        (320, 240),
        (640, 480),
        overrides=TeleopConfigOverrides(
            cube_a_colors=("green",),
            cube_b_colors=("purple",),
        ),
    )

    assert kwargs["config"].cube_a_colors == ["green"]
    assert kwargs["config"].cube_b_colors == ["purple"]


def test_recording_env_kwargs_pins_joint_control_mode() -> None:
    """Teleop records absolute joint positions, never an end-effector action.

    Joint positions are what the leader arm produces; every other action space is
    a function of them that a consumer recomputes offline. Recording one of those
    instead would bake a solver's conventions into the dataset.
    """
    kwargs = _recording_env_kwargs("MuJoCoPickLift-v1", (320, 240), (640, 480))
    assert kwargs["control_mode"] == "pd_joint_pos"
    assert RECORDING_CONTROL_MODE == "pd_joint_pos"


def test_recording_env_kwargs_overrides_a_registered_end_effector_control_mode(monkeypatch) -> None:
    """A registry or profile asking for an EE mode cannot change what gets recorded."""
    import gymnasium as gym

    spec = gym.spec("MuJoCoPickLift-v1")
    entry_point = spec.entry_point
    assert isinstance(entry_point, str)
    module_name, attr_name = entry_point.split(":")
    env_ctor = getattr(__import__(module_name, fromlist=[attr_name]), attr_name)

    monkeypatch.setattr(
        "so101_nexus.teleop.session._resolve_env_ctor",
        lambda _env_id: (env_ctor, {"control_mode": "pd_ee_pose"}),
    )

    kwargs = _recording_env_kwargs("MuJoCoPickLift-v1", (320, 240), (640, 480))
    assert kwargs["control_mode"] == "pd_joint_pos"


def test_recording_follower_declares_joint_action_features(tmp_path) -> None:
    """End of the chain: the recorder's follower writes ``<joint>.pos``, not ``ee.*``.

    The pin is only worth anything if it survives into the dataset schema, which
    is built from ``SimSOFollower.action_features``.
    """
    import pytest

    pytest.importorskip("lerobot")
    from so101_nexus.config import SO101_JOINT_NAMES
    from so101_nexus.lerobot_adapter.sim_follower import SimSOFollower
    from so101_nexus.teleop.session import build_sim_follower_config

    config = build_sim_follower_config(
        env_id="MuJoCoPickLift-v1",
        robot_id="teleop_sim",
        wrist_wh=(320, 240),
        overhead_wh=(640, 480),
        calibration_dir=tmp_path,
    )
    follower = SimSOFollower(config)
    assert follower.action_features == {f"{name}.pos": float for name in SO101_JOINT_NAMES}
