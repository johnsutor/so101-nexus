"""Tests for so101_nexus_core.teleop.leader.

`get_leader` dynamically imports lerobot; we patch the specific import targets
instead of requiring lerobot in the core test group.
"""

from __future__ import annotations

import sys
import types

import pytest

from so101_nexus_core.teleop.leader import (
    DEFAULT_WRIST_ROLL_OFFSET_DEG,
    ROBOT_JOINT_NAMES,
    check_robot_env_mismatch,
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
    assert check_robot_env_mismatch("ManiSkillReachSO101-v1", "so100") is not None


def test_check_robot_env_mismatch_matching_pair():
    assert check_robot_env_mismatch("MuJoCoReach-v1", "so101") is None


def test_check_robot_env_mismatch_so101_with_so100_env():
    assert check_robot_env_mismatch("ManiSkillReachSO100-v1", "so101") is not None


def test_import_backend_for_env_id_mujoco():
    pytest.importorskip("so101_nexus_mujoco")
    import_backend_for_env_id("MuJoCoReach-v1")
    assert "so101_nexus_mujoco" in sys.modules


def test_import_backend_for_env_id_rejects_unknown():
    with pytest.raises(ValueError):
        import_backend_for_env_id("OtherBackendEnv-v1")
