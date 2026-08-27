"""GSO (Google Scanned Objects) mesh asset management - downloads from HuggingFace on demand.

The cache-check/download/decompose/manifest orchestration is source-agnostic
and lives in ``so101_nexus.mesh_assets`` as ``ensure_scanned_mesh_assets``,
shared with ``so101_nexus.ycb_assets``. This module supplies only what is
specific to the GSO mirror: the repository id, the cache directory, and the
``fetch``/``ensure_texture`` hooks.

Unlike ai-habitat/ycb, the mirrored GSO scans (from
https://github.com/kevinzakka/mujoco_scanned_objects, itself mirroring Google
Scanned Objects, CC-BY-4.0) already ship real-world-scale OBJ + PNG, so there
is no GLB conversion or embedded-texture extraction step - the downloaded
files are used directly as ``visual.obj`` / ``texture.png``.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

from so101_nexus.constants import GSO_OBJECTS
from so101_nexus.mesh_assets import (
    _COLLISION_SUBDIR,
    MeshCollisionPart,
    ensure_scanned_mesh_assets,
    read_collision_parts,
)

logger = logging.getLogger(__name__)

_HF_REPO_ID = os.environ.get("SO101_GSO_HF_REPO", "johnsutor/gso-so101-nexus")
_CACHE_DIR = Path.home() / ".cache" / "so101_nexus" / "gso"

# Backward-compatible-style public alias, mirroring ``YCBCollisionPart``; the
# dataclass itself lives in ``mesh_assets`` so YCB and GSO share one type.
GSOCollisionPart = MeshCollisionPart


def _validate_model_id(model_id: str) -> None:
    if model_id not in GSO_OBJECTS:
        raise ValueError(f"model_id must be one of {list(GSO_OBJECTS)}, got {model_id!r}")


def get_gso_mesh_dir(model_id: str) -> Path:
    """Return the local cache directory for a GSO model's mesh files."""
    _validate_model_id(model_id)
    return _CACHE_DIR / model_id


def get_gso_texture_file(model_id: str) -> Path:
    """Return the expected local cache path for a GSO model's texture image."""
    _validate_model_id(model_id)
    return _CACHE_DIR / model_id / "texture.png"


def _collision_dir(model_id: str) -> Path:
    """Return the versioned directory holding a model's convex collision parts."""
    return _CACHE_DIR / model_id / _COLLISION_SUBDIR


def ensure_gso_assets(model_id: str) -> Path:
    """Download GSO mesh assets from HuggingFace if not already cached.

    Downloads the mirrored ``meshes/{model_id}/model.obj`` and ``texture.png``
    (already real-world scale, so no conversion is needed). A deterministic
    surface-error gate keeps one collision hull when it fits the scan; other
    objects use a cached CoACD decomposition, so physics sees concavities that
    one hull fills - the same pipeline ``ensure_ycb_assets`` uses.

    Decomposition needs the optional ``decomp`` extra. Without it, collision
    geometry falls back to one convex hull.

    Returns the directory containing the model's mesh files.
    """
    _validate_model_id(model_id)
    mesh_dir = _CACHE_DIR / model_id
    collision_dir = _collision_dir(model_id)
    visual_path = mesh_dir / "visual.obj"
    texture_path = mesh_dir / "texture.png"
    downloaded_obj = _CACHE_DIR / "meshes" / model_id / "model.obj"
    downloaded_texture = _CACHE_DIR / "meshes" / model_id / "texture.png"

    def _snapshot_download() -> None:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=_HF_REPO_ID,
            repo_type="dataset",
            allow_patterns=[f"meshes/{model_id}/*"],
            local_dir=str(_CACHE_DIR),
        )

    def _ensure_texture() -> None:
        if not downloaded_texture.exists():
            _snapshot_download()
        if downloaded_texture.exists():
            texture_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(downloaded_texture, texture_path)
        else:
            logger.warning(
                "GSO mirror has no texture.png for %r; object will render in "
                "MuJoCo's default gray.",
                model_id,
            )

    def _fetch() -> None:
        _snapshot_download()
        mesh_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(downloaded_obj, visual_path)
        _ensure_texture()

    return ensure_scanned_mesh_assets(
        model_id=model_id,
        mesh_dir=mesh_dir,
        collision_dir=collision_dir,
        visual_path=visual_path,
        texture_path=texture_path,
        fetch=_fetch,
        ensure_texture=_ensure_texture,
    )


def get_gso_collision_parts(model_id: str) -> list[MeshCollisionPart]:
    """Return the cached convex collision parts of a GSO model, in export order.

    Raises
    ------
    FileNotFoundError
        If the model has not been prepared yet; call
        :func:`ensure_gso_assets` first.
    """
    _validate_model_id(model_id)
    return read_collision_parts(_collision_dir(model_id), f"ensure_gso_assets({model_id!r})")


def get_gso_collision_meshes(model_id: str) -> list[Path]:
    """Return the OBJ paths of a GSO model's convex collision parts.

    Raises
    ------
    FileNotFoundError
        If the model has not been prepared yet; call
        :func:`ensure_gso_assets` first.
    """
    return [part.path for part in get_gso_collision_parts(model_id)]


def get_gso_collision_mesh(model_id: str) -> Path:
    """Return the path to the first convex part of a GSO model's collision mesh.

    Raises
    ------
    FileNotFoundError
        If the model has not been prepared yet; call
        :func:`ensure_gso_assets` first.
    """
    return get_gso_collision_parts(model_id)[0].path


def get_gso_visual_mesh(model_id: str) -> Path:
    """Return the path to the visual mesh for a GSO model."""
    _validate_model_id(model_id)
    return _CACHE_DIR / model_id / "visual.obj"


__all__ = [
    "GSOCollisionPart",
    "ensure_gso_assets",
    "get_gso_collision_mesh",
    "get_gso_collision_meshes",
    "get_gso_collision_parts",
    "get_gso_mesh_dir",
    "get_gso_texture_file",
    "get_gso_visual_mesh",
]
