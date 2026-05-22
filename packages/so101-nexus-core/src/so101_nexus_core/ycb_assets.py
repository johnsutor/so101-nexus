"""YCB mesh asset management — downloads from HuggingFace on demand."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Protocol, cast

from so101_nexus_core.constants import YCB_OBJECTS

logger = logging.getLogger(__name__)

_HF_REPO_ID = os.environ.get("SO101_YCB_HF_REPO", "ai-habitat/ycb")
_CACHE_DIR = Path.home() / ".cache" / "so101_nexus" / "ycb"


class _ExportableMesh(Protocol):
    def export(self, file_obj: str, file_type: str | None = None) -> object: ...

    @property
    def convex_hull(self) -> _ExportableMesh: ...


class _TextureImage(Protocol):
    def save(self, fp: str, format: str | None = None) -> object: ...


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


def _load_exportable_mesh(glb_path: Path) -> _ExportableMesh:
    """Load a GLB as a mesh object with `export` and `convex_hull`."""
    import trimesh

    scene_or_mesh = trimesh.load(str(glb_path), force="mesh")
    if isinstance(scene_or_mesh, trimesh.Scene):
        return cast("_ExportableMesh", scene_or_mesh.dump(concatenate=True))
    return cast("_ExportableMesh", scene_or_mesh)


def _convert_glb_to_obj(glb_path: Path, obj_path: Path) -> None:
    """Convert a GLB mesh to OBJ format using trimesh."""
    mesh = _load_exportable_mesh(glb_path)
    mesh.export(str(obj_path), file_type="obj")


def _texture_image_from_material(material: object) -> _TextureImage | None:
    for attr in ("image", "baseColorTexture"):
        value = getattr(material, attr, None)
        if value is None:
            continue
        image = getattr(value, "image", value)
        if image is not None:
            return cast("_TextureImage", image)
    return None


def _iter_texture_meshes(scene_or_mesh: object, scene_type: type) -> Iterator[object]:
    if isinstance(scene_or_mesh, scene_type):
        geometry = getattr(scene_or_mesh, "geometry", None)
        if isinstance(geometry, Mapping):
            yield from geometry.values()
        to_geometry = getattr(scene_or_mesh, "to_geometry", None)
        if callable(to_geometry):
            yield to_geometry()
        elif callable(dump := getattr(scene_or_mesh, "dump", None)):
            yield dump(concatenate=True)
    else:
        yield scene_or_mesh


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


def ensure_ycb_assets(model_id: str) -> Path:
    """Download YCB mesh assets from HuggingFace if not already cached.

    Downloads from the ai-habitat/ycb dataset and converts GLB meshes
    to OBJ format for use with MuJoCo and ManiSkill. The visual mesh is
    exported directly and the collision mesh is derived from its convex hull.

    Returns the directory containing the model's mesh files.
    """
    _validate_model_id(model_id)
    mesh_dir = _CACHE_DIR / model_id

    collision_path = mesh_dir / "collision.obj"
    visual_path = mesh_dir / "visual.obj"
    texture_path = mesh_dir / "texture.png"
    glb_path = _CACHE_DIR / "meshes" / model_id / "google_16k" / "textured.glb"
    orig_path = glb_path.with_suffix(".glb.orig")

    if collision_path.exists() and visual_path.exists():
        if not texture_path.exists():
            if not glb_path.exists() or not orig_path.exists():
                from huggingface_hub import snapshot_download

                snapshot_download(
                    repo_id=_HF_REPO_ID,
                    repo_type="dataset",
                    allow_patterns=[f"meshes/{model_id}/*"],
                    local_dir=str(_CACHE_DIR),
                )
            preferred_glb = _texture_glb_path(model_id)
            if preferred_glb.exists():
                extracted = _extract_glb_texture(preferred_glb, texture_path)
                if not extracted:
                    logger.warning(
                        "Failed to extract texture for YCB %r from %s; "
                        "object will render in MuJoCo's default gray.",
                        model_id,
                        preferred_glb,
                    )
        return mesh_dir

    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=_HF_REPO_ID,
        repo_type="dataset",
        allow_patterns=[f"meshes/{model_id}/*"],
        local_dir=str(_CACHE_DIR),
    )

    mesh_dir.mkdir(parents=True, exist_ok=True)
    _convert_glb_to_obj(glb_path, visual_path)

    mesh = _load_exportable_mesh(glb_path)
    hull = mesh.convex_hull
    hull.export(str(collision_path), file_type="obj")
    preferred_glb = _texture_glb_path(model_id)
    extracted = _extract_glb_texture(preferred_glb, texture_path)
    if not extracted:
        logger.warning(
            "Failed to extract texture for YCB %r from %s; "
            "object will render in MuJoCo's default gray.",
            model_id,
            preferred_glb,
        )

    return mesh_dir


def get_ycb_collision_mesh(model_id: str) -> Path:
    """Return the path to the collision mesh for a YCB model."""
    _validate_model_id(model_id)
    return _CACHE_DIR / model_id / "collision.obj"


def get_ycb_visual_mesh(model_id: str) -> Path:
    """Return the path to the visual mesh for a YCB model."""
    _validate_model_id(model_id)
    return _CACHE_DIR / model_id / "visual.obj"
