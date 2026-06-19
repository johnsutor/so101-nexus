"""Backend-dependent tests for teleop env-kwarg resolution.

These live in the mujoco package because they exercise
``_recording_env_kwargs`` against real registered MuJoCo envs.
"""

from __future__ import annotations

import so101_nexus.mujoco  # noqa: F401 - registers gym envs
from so101_nexus.objects import CubeObject, YCBObject
from so101_nexus.observations import OverheadCamera, WristCamera
from so101_nexus.teleop.config_customization import TeleopConfigOverrides
from so101_nexus.teleop.session import _recording_env_kwargs


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
            object_specs=("cube:green", "ycb:011_banana"),
            n_distractors=1,
        ),
    )

    assert kwargs["config"].n_distractors == 1
    assert isinstance(kwargs["config"].objects[0], CubeObject)
    assert kwargs["config"].objects[0].color == "green"
    assert isinstance(kwargs["config"].objects[1], YCBObject)
    assert kwargs["config"].objects[1].model_id == "011_banana"
