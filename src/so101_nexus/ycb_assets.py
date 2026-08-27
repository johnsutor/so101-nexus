"""YCB mesh asset management - downloads from HuggingFace on demand.

The cache-check/download/decompose/manifest orchestration is source-agnostic
and lives in ``so101_nexus.mesh_assets`` as ``ensure_scanned_mesh_assets``,
shared with ``so101_nexus.gso_assets``. This module supplies only what is
specific to the ai-habitat/ycb layout: the repository id, the cache directory,
and the ``fetch``/``ensure_texture`` hooks that download ``textured.glb``
scans, convert them to OBJ, and extract the embedded texture.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from so101_nexus.constants import YCB_OBJECTS
from so101_nexus.mesh_assets import (
    _COACD_SETTINGS,  # noqa: F401  (re-exported: patched by tests/core/test_ycb.py)
    _FALLBACK_DECOMPOSER,  # noqa: F401  (re-exported: patched by tests/core/test_ycb.py)
    _HULL_DECOMPOSER,  # noqa: F401  (re-exported: patched by tests/core/test_ycb.py)
    _HULL_GAP_GATE_M,  # noqa: F401  (re-exported: patched by tests/core/test_ycb.py)
    _HULL_GAP_MAX_GATE_M,  # noqa: F401  (re-exported: patched by tests/core/test_ycb.py)
    _HULL_GAP_SAMPLES,  # noqa: F401  (re-exported: patched by tests/core/test_ycb.py)
    MeshCollisionPart,
    _build_collision_geometry,  # noqa: F401  (re-exported: patched by tests/core/test_ycb.py)
    _collision_parts_are_current,  # noqa: F401  (re-exported: patched by tests/core/test_ycb.py)
    _decomposer_id,  # noqa: F401  (re-exported: patched by tests/core/test_ycb.py)
    _iter_texture_meshes,
    _load_exportable_mesh,
    _read_manifest,  # noqa: F401  (re-exported: patched by tests/core/test_ycb.py)
    _sha256,  # noqa: F401  (re-exported: patched by tests/core/test_ycb.py)
    _texture_image_from_material,
    _write_collision_parts,  # noqa: F401  (re-exported: called directly by tests/core/test_ycb.py)
    ensure_scanned_mesh_assets,
    read_collision_parts,
)

logger = logging.getLogger(__name__)

_HF_REPO_ID = os.environ.get("SO101_YCB_HF_REPO", "ai-habitat/ycb")
_CACHE_DIR = Path.home() / ".cache" / "so101_nexus" / "ycb"

# Backward-compatible public alias; the dataclass itself now lives in
# ``mesh_assets`` so YCB and GSO share one type.
YCBCollisionPart = MeshCollisionPart


def _validate_model_id(model_id: str) -> None:
    if model_id not in YCB_OBJECTS:
        raise ValueError(f"model_id must be one of {list(YCB_OBJECTS)}, got {model_id!r}")


def get_ycb_mesh_dir(model_id: str) -> Path:
    """Return the local cache directory for a YCB model's mesh files."""
    _validate_model_id(model_id)
    return _CACHE_DIR / model_id


def get_ycb_texture_file(model_id: str) -> Path:
    """Return the expected local cache path for a YCB model's texture image."""
    _validate_model_id(model_id)
    return _CACHE_DIR / model_id / "texture.png"


def _convert_glb_to_obj(glb_path: Path, obj_path: Path) -> None:
    """Convert a GLB mesh to OBJ format using trimesh."""
    mesh = _load_exportable_mesh(glb_path)
    mesh.export(str(obj_path), file_type="obj")


def _collision_dir(model_id: str) -> Path:
    """Return the versioned directory holding a model's convex collision parts."""
    from so101_nexus.mesh_assets import _COLLISION_SUBDIR

    return _CACHE_DIR / model_id / _COLLISION_SUBDIR


def _extract_glb_texture(glb_path: Path, texture_path: Path) -> bool:
    """Extract the first available GLB material texture into ``texture_path``.

    ``glb_path`` may use the ``.glb.orig`` suffix used by the ai-habitat/ycb
    dataset; ``file_type="glb"`` skips trimesh's extension-based type
    inference, which raises ``NotImplementedError`` for unknown suffixes.
    """
    import trimesh

    scene_or_mesh = trimesh.load(str(glb_path), file_type="glb")
    for mesh in _iter_texture_meshes(scene_or_mesh, trimesh.Scene):
        visual = getattr(mesh, "visual", None)
        material = getattr(visual, "material", None)
        if material is None:
            continue
        image = _texture_image_from_material(material)
        if image is None:
            continue
        texture_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(str(texture_path), format="PNG")
        return True
    return False


def _texture_glb_path(model_id: str) -> Path:
    """Return the preferred GLB path for YCB texture extraction.

    The ai-habitat/ycb dataset ships ``textured.glb`` (optimized, embedded
    texture stripped) alongside ``textured.glb.orig`` (original, with the
    embedded texture). Prefer the ``.orig`` form when available so
    :func:`_extract_glb_texture` can recover the texture.
    """
    base = _CACHE_DIR / "meshes" / model_id / "google_16k"
    orig = base / "textured.glb.orig"
    return orig if orig.exists() else base / "textured.glb"


def _warn_texture_extraction_failed(model_id: str, glb_path: Path) -> None:
    logger.warning(
        "Failed to extract texture for YCB %r from %s; "
        "object will render in MuJoCo's default gray.",
        model_id,
        glb_path,
    )


def ensure_ycb_assets(model_id: str) -> Path:
    """Download YCB mesh assets from HuggingFace if not already cached.

    Downloads the ai-habitat/ycb meshes and converts them to OBJ for MuJoCo.
    The visual mesh stays unchanged. A deterministic surface-error gate keeps
    one collision hull when it fits the scan. Other objects use a cached CoACD
    decomposition, so physics sees concavities that one hull fills.

    Decomposition needs the optional ``decomp`` extra. Without it, collision
    geometry falls back to one convex hull.

    Returns the directory containing the model's mesh files.
    """
    _validate_model_id(model_id)
    mesh_dir = _CACHE_DIR / model_id
    collision_dir = _collision_dir(model_id)
    visual_path = mesh_dir / "visual.obj"
    texture_path = mesh_dir / "texture.png"
    glb_path = _CACHE_DIR / "meshes" / model_id / "google_16k" / "textured.glb"
    orig_path = glb_path.with_suffix(".glb.orig")

    def _snapshot_download() -> None:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=_HF_REPO_ID,
            repo_type="dataset",
            allow_patterns=[f"meshes/{model_id}/*"],
            local_dir=str(_CACHE_DIR),
        )

    def _ensure_texture() -> None:
        if not glb_path.exists() or not orig_path.exists():
            _snapshot_download()
        preferred_glb = _texture_glb_path(model_id)
        if preferred_glb.exists() and not _extract_glb_texture(preferred_glb, texture_path):
            _warn_texture_extraction_failed(model_id, preferred_glb)

    def _fetch() -> None:
        _snapshot_download()
        mesh_dir.mkdir(parents=True, exist_ok=True)
        _convert_glb_to_obj(glb_path, visual_path)
        # Single-hull cache from a release before the decomposition; nothing reads it.
        (mesh_dir / "collision.obj").unlink(missing_ok=True)
        preferred_glb = _texture_glb_path(model_id)
        if not _extract_glb_texture(preferred_glb, texture_path):
            _warn_texture_extraction_failed(model_id, preferred_glb)

    return ensure_scanned_mesh_assets(
        model_id=model_id,
        mesh_dir=mesh_dir,
        collision_dir=collision_dir,
        visual_path=visual_path,
        texture_path=texture_path,
        fetch=_fetch,
        ensure_texture=_ensure_texture,
    )


def get_ycb_collision_parts(model_id: str) -> list[MeshCollisionPart]:
    """Return the cached convex collision parts of a YCB model, in export order.

    Raises
    ------
    FileNotFoundError
        If the model has not been prepared yet; call
        :func:`ensure_ycb_assets` first.
    """
    _validate_model_id(model_id)
    return read_collision_parts(_collision_dir(model_id), f"ensure_ycb_assets({model_id!r})")


def get_ycb_collision_meshes(model_id: str) -> list[Path]:
    """Return the OBJ paths of a YCB model's convex collision parts.

    Raises
    ------
    FileNotFoundError
        If the model has not been prepared yet; call
        :func:`ensure_ycb_assets` first.
    """
    return [part.path for part in get_ycb_collision_parts(model_id)]


def get_ycb_collision_mesh(model_id: str) -> Path:
    """Return the path to the first convex part of a YCB model's collision mesh.

    Raises
    ------
    FileNotFoundError
        If the model has not been prepared yet; call
        :func:`ensure_ycb_assets` first.
    """
    return get_ycb_collision_parts(model_id)[0].path


def get_ycb_visual_mesh(model_id: str) -> Path:
    """Return the path to the visual mesh for a YCB model."""
    _validate_model_id(model_id)
    return _CACHE_DIR / model_id / "visual.obj"
