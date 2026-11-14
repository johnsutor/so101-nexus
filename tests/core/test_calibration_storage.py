"""Physical calibration belongs in durable per-user storage, independent of launch paths."""

import sys
from dataclasses import replace

import pytest

from so101_nexus import SO101_JOINT_NAMES, RobotProfile


@pytest.fixture(autouse=True)
def storage_environment(monkeypatch, tmp_path):
    for name in ("SO101_NEXUS_DATA_DIR", "SO101_NEXUS_PROFILES_DIR", "SO101_NEXUS_CALIBRATION_DIR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))


@pytest.fixture
def profile():
    return RobotProfile(
        "storage-arm",
        {
            name: {"id": i, "drive_mode": 0, "homing_offset": 0, "range_min": 0, "range_max": 4095}
            for i, name in enumerate(SO101_JOINT_NAMES, 1)
        },
    )


def test_platform_storage_honors_data_environment_and_ignores_cache(tmp_path):
    from platformdirs import user_data_path

    from so101_nexus.calibration.storage import calibration_dir, profiles_dir

    root = user_data_path("so101-nexus", appauthor=False)
    assert profiles_dir() == root / "profiles"
    assert calibration_dir() == root / "calibration"
    assert not root.exists()
    assert tmp_path / "cache" not in profiles_dir().parents


def test_explicit_directory_overrides_are_independent(monkeypatch, tmp_path):
    from so101_nexus.calibration.storage import calibration_dir, profiles_dir

    monkeypatch.setenv("SO101_NEXUS_DATA_DIR", str(tmp_path / "custom"))
    assert profiles_dir() == tmp_path / "custom/profiles"
    monkeypatch.setenv("SO101_NEXUS_PROFILES_DIR", str(tmp_path / "robots"))
    monkeypatch.setenv("SO101_NEXUS_CALIBRATION_DIR", str(tmp_path / "experiments"))
    assert profiles_dir() == tmp_path / "robots"
    assert calibration_dir() == tmp_path / "experiments"


def test_import_is_durable_idempotent_and_preserves_conflicts(profile, tmp_path, monkeypatch):
    from so101_nexus.calibration.storage import import_profile, profiles_dir, saved_profiles

    source = tmp_path / "upload/robot.json"
    profile.save(source)
    saved = import_profile(source)
    assert saved.parent == profiles_dir()
    assert import_profile(source) == saved
    other = replace(profile, joint_offsets_deg=(1, 2, 3, 4, 5))
    other.save(source)
    second = import_profile(source)
    assert second != saved
    assert import_profile(source) == second
    assert RobotProfile.load(saved) == profile
    assert RobotProfile.load(second) == other
    source.unlink()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert set(saved_profiles()) == {saved, second}


def test_invalid_import_does_not_create_a_profile(tmp_path):
    from so101_nexus.calibration.storage import import_profile, profiles_dir

    source = tmp_path / "invalid.json"
    source.write_text("{}")
    with pytest.raises(ValueError, match="schema_version"):
        import_profile(source)
    assert not profiles_dir().exists()


def test_legacy_migration_preserves_full_snapshot_and_originals(profile, tmp_path):
    from so101_nexus.calibration.storage import calibration_dir, migrate_legacy, profiles_dir

    legacy = tmp_path / "old/runs/calibration"
    source = legacy / "robot.json"
    profile.save(source)
    recorded = legacy / "recordings/00_multisine.npz"
    recorded.parent.mkdir()
    recorded.write_bytes(b"recording")
    fitted = legacy / "fit/candidate_profile.json"
    profile.save(fitted)
    (legacy / "bad.json").write_text("{}")
    original = {p.relative_to(legacy): p.read_bytes() for p in legacy.rglob("*") if p.is_file()}
    imported = migrate_legacy(legacy)
    assert len(imported) == 1
    assert imported[0].parent == profiles_dir()
    assert RobotProfile.load(imported[0]) == profile
    archives = list((calibration_dir() / "legacy").iterdir())
    assert len(archives) == 1
    assert {
        p.relative_to(archives[0]): p.read_bytes() for p in archives[0].rglob("*") if p.is_file()
    } == original
    assert {
        p.relative_to(legacy): p.read_bytes() for p in legacy.rglob("*") if p.is_file()
    } == original
    assert migrate_legacy(legacy) == imported
    assert list((calibration_dir() / "legacy").iterdir()) == archives
    recorded.write_bytes(b"new recording")
    migrate_legacy(legacy)
    assert len(list((calibration_dir() / "legacy").iterdir())) == 2
    assert (archives[0] / "recordings/00_multisine.npz").read_bytes() == b"recording"


def test_migration_rejects_missing_or_recursive_source(tmp_path, monkeypatch):
    from so101_nexus.calibration.storage import migrate_legacy

    with pytest.raises(ValueError, match="directory"):
        migrate_legacy(tmp_path / "missing")
    monkeypatch.setenv("SO101_NEXUS_DATA_DIR", str(tmp_path / "old/inside"))
    (tmp_path / "old").mkdir()
    with pytest.raises(ValueError, match="inside"):
        migrate_legacy(tmp_path / "old")


@pytest.mark.skipif(sys.platform != "linux", reason="Linux XDG default")
def test_linux_default_is_home_data_not_cache(monkeypatch, tmp_path):
    from so101_nexus.calibration.storage import calibration_dir, profiles_dir

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    assert profiles_dir() == tmp_path / ".local/share/so101-nexus/profiles"
    assert calibration_dir() == tmp_path / ".local/share/so101-nexus/calibration"


@pytest.mark.parametrize(
    "name", ["SO101_NEXUS_DATA_DIR", "SO101_NEXUS_PROFILES_DIR", "SO101_NEXUS_CALIBRATION_DIR"]
)
def test_relative_override_cannot_reintroduce_launch_directory_dependence(monkeypatch, name):
    from so101_nexus.calibration.storage import calibration_dir, profiles_dir

    monkeypatch.setenv(name, "relative/data")
    with pytest.raises(ValueError, match="absolute"):
        profiles_dir() if name != "SO101_NEXUS_CALIBRATION_DIR" else calibration_dir()


def test_legacy_profile_order_and_conflicts_are_preserved(profile, tmp_path):
    import os

    from so101_nexus.calibration.storage import import_profile, migrate_legacy, saved_profiles

    legacy = tmp_path / "legacy"
    newest, oldest = legacy / "newest.json", legacy / "oldest.json"
    profile.save(newest)
    replace(profile, joint_offsets_deg=(1, 2, 3, 4, 5)).save(oldest)
    os.utime(oldest, (1000, 1000))
    os.utime(newest, (2000, 2000))
    migrate_legacy(legacy)
    assert saved_profiles()[0].name == "newest.json"
    external = tmp_path / "newest.json"
    replace(profile, control_period_s=0.03).save(external)
    changed = import_profile(external)
    assert changed.name != newest.name
    assert saved_profiles()[0] == changed
    migrate_legacy(legacy)
    assert saved_profiles()[0] == changed


def test_source_checkout_legacy_is_found_from_another_directory(tmp_path, monkeypatch):
    from so101_nexus.calibration import storage

    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / "pyproject.toml").write_text("[project]\nname='so101-nexus'\n")
    monkeypatch.setattr(
        storage, "__file__", str(checkout / "src/so101_nexus/calibration/storage.py")
    )
    monkeypatch.chdir(tmp_path)
    assert checkout / "runs/calibration" in storage.legacy_directories()


def test_concurrent_import_cannot_overwrite_a_competing_profile(profile, tmp_path, monkeypatch):
    from so101_nexus.calibration import storage

    source = tmp_path / "robot.json"
    profile.save(source)
    other = replace(profile, joint_offsets_deg=(1, 2, 3, 4, 5))
    link = storage.os.link
    competed = False

    def competing_link(staging, destination):
        nonlocal competed
        if not competed:
            competed = True
            other.save(destination)
        link(staging, destination)

    monkeypatch.setattr(storage.os, "link", competing_link)
    saved = storage.import_profile(source)
    assert RobotProfile.load(storage.profiles_dir() / "robot.json") == other
    assert RobotProfile.load(saved) == profile
    assert saved.name == "robot_002.json"
    assert not list(storage.profiles_dir().glob(".import-*"))
