"""Durable per-user robot profiles and calibration datasets, separate from caches."""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from platformdirs import user_data_path

from so101_nexus.profiles import RobotProfile


def _directory(override: str, default: Path) -> Path:
    value = os.environ.get(override)
    if not value:
        return default
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{override} must be an absolute directory path.")
    return path.resolve()


def _data_dir() -> Path:
    return _directory("SO101_NEXUS_DATA_DIR", user_data_path("so101-nexus", appauthor=False))


def profiles_dir() -> Path:
    """Return the persistent profile directory without creating it."""
    return _directory("SO101_NEXUS_PROFILES_DIR", _data_dir() / "profiles")


def calibration_dir() -> Path:
    """Return the persistent recording and fit directory without creating it."""
    return _directory("SO101_NEXUS_CALIBRATION_DIR", _data_dir() / "calibration")


def saved_profiles(directory: Path | None = None) -> list[Path]:
    """List valid saved profiles with the most recently saved profile first."""
    profiles = []
    for path in (directory or profiles_dir()).glob("*.json"):
        try:
            RobotProfile.load(path)
            profiles.append(path.resolve())
        except (OSError, ValueError, TypeError, KeyError):
            continue
    return sorted(profiles, key=lambda path: path.stat().st_mtime_ns, reverse=True)


def import_profile(source: str | Path) -> Path:
    """Validate and copy a transferred profile, preserving existing versions."""
    source = Path(source).expanduser().resolve()
    profile = RobotProfile.load(source)
    directory = profiles_dir()
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{source.stem}.json"
    index = 2
    with TemporaryDirectory(prefix=".import-", dir=directory) as temporary:
        staging = Path(temporary) / "profile.json"
        profile.save(staging)
        while True:
            try:
                os.link(staging, destination)
                return destination
            except FileExistsError:
                try:
                    if RobotProfile.load(destination) == profile:
                        return destination
                except (OSError, ValueError, TypeError, KeyError):
                    pass
                destination = directory / f"{source.stem}_{index:03d}.json"
                index += 1


def legacy_directories() -> list[Path]:
    """Find legacy data in the launch directory or a source checkout."""
    roots = [Path.cwd()]
    checkout = Path(__file__).resolve().parents[3]
    if (checkout / "pyproject.toml").is_file():
        roots.append(checkout)
    return list(dict.fromkeys(root / "runs/calibration" for root in roots))


def migrate_legacy(source: str | Path) -> list[Path]:
    """Copy a complete legacy snapshot and its profiles without modifying originals."""
    source = Path(source).expanduser().resolve()
    if not source.is_dir():
        raise ValueError("Choose an existing calibration directory.")
    if any(directory.is_relative_to(source) for directory in (calibration_dir(), profiles_dir())):
        raise ValueError("Application storage must not be inside the legacy directory.")
    paths = sorted(source.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise ValueError("Import a calibration directory containing regular files, without links.")
    files = [path for path in paths if path.is_file()]
    if not files:
        return []
    destination = _archive_files(source, files)
    existing = set(saved_profiles())
    imported = []
    for path in saved_profiles(destination):
        saved = import_profile(path)
        if saved not in existing:
            shutil.copystat(path, saved)
            existing.add(saved)
        imported.append(saved)
    return imported


def _archive_files(source: Path, files: list[Path]) -> Path:
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(source).as_posix().encode() + b"\0")
        with path.open("rb") as stream:
            digest.update(hashlib.file_digest(stream, "sha256").digest())
    archives = calibration_dir() / "legacy"
    destination = archives / digest.hexdigest()
    if not destination.exists():
        archives.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".migration-", dir=archives) as temporary:
            staging = Path(temporary) / "snapshot"
            staging.mkdir()
            for path in files:
                target = staging / path.relative_to(source)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
            try:
                staging.rename(destination)
            except OSError:
                if not destination.is_dir():
                    raise
    return destination
