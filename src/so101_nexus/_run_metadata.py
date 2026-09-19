"""Runtime provenance for reproducible training artifacts."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import torch


def _package_versions() -> dict[str, str | None]:
    packages = ("so101-nexus", "numpy", "torch", "mujoco", "mujoco-warp", "warp-lang")
    found: dict[str, str | None] = {}
    for package in packages:
        try:
            found[package] = version(package)
        except PackageNotFoundError:
            found[package] = None
    found["python"] = platform.python_version()
    return found


def _git_state() -> dict[str, str | bool] | None:
    root = Path(__file__).resolve().parents[2]
    if not (root / ".git").exists():
        return None
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return {"sha": sha, "dirty": dirty}


def _stable_value(value: Any) -> Any:
    model_id = getattr(value, "model_id", None)
    if isinstance(model_id, str):
        stable = {
            "__type__": type(value).__name__,
            "model_id": model_id,
            "mass_override": getattr(value, "mass_override", None),
        }
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        stable = {
            "__type__": type(value).__name__,
            **{
                field.name: _stable_value(getattr(value, field.name))
                for field in dataclasses.fields(value)
            },
        }
    elif hasattr(value, "__dict__"):
        stable = {
            "__type__": type(value).__name__,
            **{
                key: _stable_value(item)
                for key, item in sorted(vars(value).items())
                if not key.startswith("_")
            },
        }
    elif isinstance(value, dict):
        stable = {str(key): _stable_value(item) for key, item in value.items()}
    elif isinstance(value, (list, tuple)):
        stable = [_stable_value(item) for item in value]
    elif isinstance(value, (str, int, float, bool, type(None))):
        stable = value
    else:
        stable = repr(value)
    return stable


def _asset_manifests(config: Any) -> list[dict[str, str]]:
    manifests: list[dict[str, str]] = []
    seen: set[str] = set()

    def mappings(value: Any):
        if isinstance(value, dict):
            yield value
            for item in value.values():
                yield from mappings(item)
        elif isinstance(value, list):
            for item in value:
                yield from mappings(item)

    for value in mappings(_stable_value(config)):
        model_id = value.get("model_id")
        object_type = value.get("__type__")
        if not isinstance(model_id, str):
            continue
        if object_type == "YCBObject":
            from so101_nexus.ycb_assets import _collision_dir
        elif object_type == "GSOObject":
            from so101_nexus.gso_assets import _collision_dir
        else:
            continue
        path = _collision_dir(model_id) / "manifest.json"
        if str(path) in seen or not path.is_file():
            continue
        seen.add(str(path))
        manifests.append(
            {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )

    return sorted(manifests, key=lambda item: item["path"])


def training_metadata(*, device: torch.device, env_config: Any, **values: Any) -> dict[str, Any]:
    """Return self-contained run provenance for an inference snapshot."""
    hardware: dict[str, Any] = {
        "device": str(device),
        "platform": platform.platform(),
        "cuda_runtime": torch.version.cuda,
    }
    if device.type == "cuda" and torch.cuda.is_available():
        index = device.index if device.index is not None else torch.cuda.current_device()
        hardware.update(
            name=torch.cuda.get_device_name(index),
            capability=list(torch.cuda.get_device_capability(index)),
        )
        try:
            hardware["driver"] = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.splitlines()[index]
        except (OSError, subprocess.CalledProcessError, IndexError):
            hardware["driver"] = None
    metadata = _stable_value(
        {
            **values,
            "versions": _package_versions(),
            "git": _git_state(),
            "hardware": hardware,
            "env_config": _stable_value(env_config),
            "asset_manifests": _asset_manifests(env_config),
        }
    )
    json.dumps(metadata, sort_keys=True)
    return metadata
