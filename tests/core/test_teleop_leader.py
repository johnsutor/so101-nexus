"""Tests for so101_nexus.teleop.leader.

`get_leader` dynamically imports lerobot; we patch the specific import targets
instead of requiring lerobot in the core test group.
"""

from __future__ import annotations

import sys
import types

import pytest

from so101_nexus.teleop.leader import (
    DEFAULT_WRIST_ROLL_OFFSET_DEG,
    ROBOT_JOINT_NAMES,
    check_robot_env_mismatch,
    diagnose_leader_port,
    format_leader_connection_error,
    get_leader,
    import_backend_for_env_id,
)


def _install_fake_lerobot(monkeypatch):
    """Install dummy lerobot.teleoperators.so_leader.* modules for get_leader."""
    seen: dict = {}

    class FakeConfig:
        def __init__(self, *, port, use_degrees, id):
            seen["config"] = (port, use_degrees, id)

    class FakeLeader:
        def __init__(self, config):
            seen["config_obj"] = config

    config_mod = types.ModuleType("lerobot.teleoperators.so_leader.config_so_leader")
    config_mod.SO100LeaderConfig = FakeConfig
    config_mod.SO101LeaderConfig = FakeConfig

    leader_mod = types.ModuleType("lerobot.teleoperators.so_leader.so_leader")
    leader_mod.SO100Leader = FakeLeader
    leader_mod.SO101Leader = FakeLeader

    for name, mod in [
        ("lerobot", types.ModuleType("lerobot")),
        ("lerobot.teleoperators", types.ModuleType("lerobot.teleoperators")),
        ("lerobot.teleoperators.so_leader", types.ModuleType("lerobot.teleoperators.so_leader")),
        ("lerobot.teleoperators.so_leader.config_so_leader", config_mod),
        ("lerobot.teleoperators.so_leader.so_leader", leader_mod),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)

    return seen


def test_default_wrist_roll_offset_is_negative_90():
    assert DEFAULT_WRIST_ROLL_OFFSET_DEG == -90.0


def test_apply_wrist_roll_offset_deg_shifts_wrist_roll_only() -> None:
    from so101_nexus.teleop.leader import apply_wrist_roll_offset_deg

    action = {
        "shoulder_pan.pos": 10.0,
        "shoulder_lift.pos": 20.0,
        "elbow_flex.pos": 30.0,
        "wrist_flex.pos": 40.0,
        "wrist_roll.pos": 50.0,
        "gripper.pos": 60.0,
    }
    result = apply_wrist_roll_offset_deg(action, offset_deg=-90.0)

    assert result["wrist_roll.pos"] == pytest.approx(-40.0)
    for key in (
        "shoulder_pan.pos",
        "shoulder_lift.pos",
        "elbow_flex.pos",
        "wrist_flex.pos",
        "gripper.pos",
    ):
        assert result[key] == action[key]
    assert result is not action


def test_apply_wrist_roll_offset_deg_zero_offset_is_noop_copy() -> None:
    from so101_nexus.teleop.leader import apply_wrist_roll_offset_deg

    action = {"wrist_roll.pos": 5.0, "gripper.pos": 12.0}
    result = apply_wrist_roll_offset_deg(action, offset_deg=0.0)

    assert result == action
    assert result is not action


def test_apply_wrist_roll_offset_deg_ignores_missing_wrist_roll() -> None:
    from so101_nexus.teleop.leader import apply_wrist_roll_offset_deg

    action = {"gripper.pos": 12.0}

    assert apply_wrist_roll_offset_deg(action, offset_deg=-90.0) == action


def test_robot_joint_names_both_known():
    assert "so100" in ROBOT_JOINT_NAMES
    assert "so101" in ROBOT_JOINT_NAMES


def test_get_leader_so100(monkeypatch):
    seen = _install_fake_lerobot(monkeypatch)
    leader = get_leader("so100", port="/dev/ttyACM0", leader_id="leader_a")
    assert leader is not None
    assert seen["config"] == ("/dev/ttyACM0", True, "leader_a")


def test_get_leader_so101(monkeypatch):
    seen = _install_fake_lerobot(monkeypatch)
    leader = get_leader("so101", port="/dev/ttyACM1", leader_id="leader_b")
    assert leader is not None
    assert seen["config"] == ("/dev/ttyACM1", True, "leader_b")


def test_check_robot_env_mismatch_so100_with_so101_env():
    assert check_robot_env_mismatch("CustomReachSO101-v1", "so100") is not None


def test_check_robot_env_mismatch_matching_pair():
    assert check_robot_env_mismatch("MuJoCoReach-v1", "so101") is None


def test_check_robot_env_mismatch_so101_with_so100_env():
    assert check_robot_env_mismatch("CustomReachSO100-v1", "so101") is not None


def test_import_backend_for_env_id_mujoco():
    pytest.importorskip("so101_nexus.mujoco")
    import_backend_for_env_id("MuJoCoReach-v1")
    assert "so101_nexus.mujoco" in sys.modules


def test_import_backend_for_env_id_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown custom env_id"):
        import_backend_for_env_id("OtherBackendEnv-v1")


def test_import_backend_for_env_id_allows_registered_custom_env(monkeypatch):
    import gymnasium as gym

    class _Spec:
        pass

    monkeypatch.setattr(gym, "spec", lambda env_id: _Spec())

    import_backend_for_env_id("CustomPick-v1")


def test_import_backend_for_env_id_preserves_unexpected_spec_errors(monkeypatch):
    import gymnasium as gym

    monkeypatch.setattr(gym, "spec", lambda _env_id: (_ for _ in ()).throw(RuntimeError("boom")))

    with pytest.raises(RuntimeError, match="boom"):
        import_backend_for_env_id("CustomPick-v1")


def test_diagnose_leader_port_reports_missing_path(monkeypatch):
    monkeypatch.setattr("so101_nexus.teleop.leader.os.path.exists", lambda _path: False)

    diag = diagnose_leader_port("/dev/ttyACM9")

    assert diag.kind == "not_found"
    assert "/dev/ttyACM9" in diag.message
    assert "lerobot-find-port" in diag.recovery_hint


def test_diagnose_leader_port_reports_permission_denied(monkeypatch):
    monkeypatch.setattr("so101_nexus.teleop.leader.os.path.exists", lambda _path: True)
    monkeypatch.setattr("so101_nexus.teleop.leader.os.access", lambda _path, _mode: False)

    diag = diagnose_leader_port("/dev/ttyACM0")

    assert diag.kind == "permission_denied"
    assert "/dev/ttyACM0" in diag.message
    assert "chmod" in diag.recovery_hint


def test_format_leader_connection_error_includes_recovery_hint(monkeypatch):
    monkeypatch.setattr("so101_nexus.teleop.leader.os.path.exists", lambda _path: True)
    monkeypatch.setattr("so101_nexus.teleop.leader.os.access", lambda _path, _mode: False)

    msg = format_leader_connection_error("/dev/ttyACM0", OSError("permission denied"))

    assert "Failed to connect on /dev/ttyACM0" in msg
    assert "permission denied" in msg.lower()
    assert "chmod 666 /dev/ttyACM0" in msg
