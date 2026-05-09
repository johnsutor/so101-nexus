"""ManiSkill-specific teleop environment kwarg tests."""

from __future__ import annotations

import so101_nexus_maniskill  # noqa: F401 - registers gym envs
from so101_nexus_core.objects import CubeObject, YCBObject
from so101_nexus_core.observations import OverheadCamera, WristCamera
from so101_nexus_core.teleop.config_customization import TeleopConfigOverrides
from so101_nexus_core.teleop.session import _recording_env_kwargs


def test_recording_env_kwargs_wires_maniskill_cameras() -> None:
    """Teleop must add camera observations to ManiSkill env configs."""
    kwargs = _recording_env_kwargs("ManiSkillLookAtSO101-v1", (320, 240), (640, 480))

    observations = kwargs["config"].observations
    wrist = [o for o in observations if isinstance(o, WristCamera)]
    overhead = [o for o in observations if isinstance(o, OverheadCamera)]

    assert len(wrist) == 1
    assert wrist[0].width == 320
    assert wrist[0].height == 240
    assert len(overhead) == 1
    assert overhead[0].width == 640
    assert overhead[0].height == 480


def test_recording_env_kwargs_requests_maniskill_rgb_observations() -> None:
    """Teleop preview frames require ManiSkill RGB sensor observations."""
    kwargs = _recording_env_kwargs("ManiSkillLookAtSO101-v1", (320, 240), (640, 480))

    assert kwargs["obs_mode"] == "rgb"


def test_recording_env_kwargs_requests_maniskill_absolute_joint_control() -> None:
    """Teleop actions are absolute joint positions, matching MuJoCo teleop."""
    kwargs = _recording_env_kwargs("ManiSkillLookAtSO101-v1", (320, 240), (640, 480))

    assert kwargs["control_mode"] == "pd_joint_pos"


def test_recording_env_kwargs_preserves_maniskill_rgb_with_overrides() -> None:
    kwargs = _recording_env_kwargs(
        "ManiSkillPickLiftSO101-v1",
        (320, 240),
        (640, 480),
        overrides=TeleopConfigOverrides(object_specs=("cube:green", "ycb:011_banana")),
    )

    assert kwargs["obs_mode"] == "rgb"
    assert kwargs["control_mode"] == "pd_joint_pos"
    assert isinstance(kwargs["config"].objects[0], CubeObject)
    assert kwargs["config"].objects[0].color == "green"
    assert isinstance(kwargs["config"].objects[1], YCBObject)
    assert kwargs["config"].objects[1].model_id == "011_banana"
