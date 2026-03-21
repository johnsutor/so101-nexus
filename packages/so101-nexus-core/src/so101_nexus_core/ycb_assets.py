"""YCB mesh asset management — downloads from HuggingFace on demand."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol, cast

from so101_nexus_core.constants import YCB_OBJECTS

_HF_REPO_ID = os.environ.get("SO101_YCB_HF_REPO", "ai-habitat/ycb")
_CACHE_DIR = Path.home() / ".cache" / "so101_nexus" / "ycb"


class _ExportableMesh(Protocol):
    def export(self, file_obj: str, file_type: str | None = None) -> object: ...

    @property
    def convex_hull(self) -> "_ExportableMesh": ...


def _validate_model_id(model_id: str) -> None:
    if model_id not in YCB_OBJECTS:
        raise ValueError(f"model_id must be one of {list(YCB_OBJECTS)}, got {model_id!r}")


def get_ycb_mesh_dir(model_id: str) -> Path:
    """Return the local cache directory for a YCB model's mesh files."""
    _validate_model_id(model_id)
    return _CACHE_DIR / model_id


def _load_exportable_mesh(glb_path: Path) -> _ExportableMesh:
    """Load a GLB as a mesh object with `export` and `convex_hull`."""
    import trimesh

    scene_or_mesh = trimesh.load(str(glb_path), force="mesh")
    if isinstance(scene_or_mesh, trimesh.Scene):
        return cast(_ExportableMesh, scene_or_mesh.dump(concatenate=True))
    return cast(_ExportableMesh, scene_or_mesh)


def _convert_glb_to_obj(glb_path: Path, obj_path: Path) -> None:
    """Convert a GLB mesh to OBJ format using trimesh."""
    mesh = _load_exportable_mesh(glb_path)
    mesh.export(str(obj_path), file_type="obj")


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

    if collision_path.exists() and visual_path.exists():
        return mesh_dir

    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=_HF_REPO_ID,
        repo_type="dataset",
        allow_patterns=[f"meshes/{model_id}/*"],
        local_dir=str(_CACHE_DIR),
    )

    glb_path = _CACHE_DIR / "meshes" / model_id / "google_16k" / "textured.glb"

    mesh_dir.mkdir(parents=True, exist_ok=True)
    _convert_glb_to_obj(glb_path, visual_path)

    mesh = _load_exportable_mesh(glb_path)
    hull = mesh.convex_hull
    hull.export(str(collision_path), file_type="obj")

    return mesh_dir


def get_ycb_collision_mesh(model_id: str) -> Path:
    """Return the path to the collision mesh for a YCB model."""
    _validate_model_id(model_id)
    return _CACHE_DIR / model_id / "collision.obj"


def get_ycb_visual_mesh(model_id: str) -> Path:
    """Return the path to the visual mesh for a YCB model."""
    _validate_model_id(model_id)
    return _CACHE_DIR / model_id / "visual.obj"
