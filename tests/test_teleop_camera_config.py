from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_teleop_module():
    teleop_path = Path(__file__).resolve().parents[1] / "examples" / "teleop.py"
    spec = importlib.util.spec_from_file_location("test_examples_teleop", teleop_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_recording_env_kwargs_override_mujoco_camera_size():
    teleop = _load_teleop_module()

    kwargs = teleop._recording_env_kwargs("MuJoCoPickCubeLift-v1", 480, 512)

    assert kwargs["config"].camera.width == 480
    assert kwargs["config"].camera.height == 512


def test_recording_env_kwargs_preserve_registered_env_kwargs():
    teleop = _load_teleop_module()

    kwargs = teleop._recording_env_kwargs("MuJoCoPickGelatinBoxLift-v1", 640, 360)

    assert kwargs["model_id"] == "009_gelatin_box"
    assert kwargs["config"].camera.width == 640
    assert kwargs["config"].camera.height == 360
