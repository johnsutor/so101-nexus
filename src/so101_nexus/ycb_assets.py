"""YCB mesh asset management - downloads from HuggingFace on demand."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

from so101_nexus.constants import YCB_OBJECTS

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)

_HF_REPO_ID = os.environ.get("SO101_YCB_HF_REPO", "ai-habitat/ycb")
_CACHE_DIR = Path.home() / ".cache" / "so101_nexus" / "ycb"

_COLLISION_SUBDIR = "collision_v2"
"""Cache subdirectory holding the convex decomposition.

Versioned: bump it whenever the decomposition changes, otherwise the parts
cached by an older release keep winning the ``ensure_ycb_assets`` early return.
"""

_MANIFEST_NAME = "parts.json"

_HULL_DECOMPOSER = "convex_hull"
"""Manifest marker for parts built without CoACD (the single-hull fallback)."""

# CoACD settings. The seed is fixed because scene geometry must not vary
# between runs; preprocessing (watertight remeshing of the open YCB scans) is
# what lifts the thin, concave models above the acceptance convexity, and the
# hull cap keeps the per-object geom count affordable for contact budgets.
_COACD_SEED = 0
_COACD_THRESHOLD = 0.02
_COACD_MAX_HULLS = 16
_COACD_PREPROCESS_RESOLUTION = 100

_SINGLE_HULL_CONVEXITY = 0.95
"""Volume ratio above which a model keeps its single convex hull.

Boxes and balls are already their own hull; decomposing them only buys geoms.
"""


class _ExportableMesh(Protocol):
    vertices: np.ndarray
    faces: np.ndarray

    @property
    def volume(self) -> float: ...

    def export(self, file_obj: str, file_type: str | None = None) -> object: ...

    @property
    def convex_hull(self) -> _ExportableMesh: ...


class _TextureImage(Protocol):
    def save(self, fp: str, format: str | None = None) -> object: ...


@dataclass(frozen=True, slots=True)
class YCBCollisionPart:
    """One convex part of a YCB model's collision decomposition.

    Attributes
    ----------
    path:
        OBJ file holding the convex part.
    mass_fraction:
        Share of the object's mass carried by this part, proportional to its
        volume. A model's fractions sum to 1, so the body's total mass does not
        depend on how many parts the decomposition produced.
    """

    path: Path
    mass_fraction: float


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


def _collision_dir(model_id: str) -> Path:
    """Return the versioned directory holding a model's convex collision parts."""
    return _CACHE_DIR / model_id / _COLLISION_SUBDIR


def _decomposer_id() -> str:
    """Identify the decomposer that would produce the parts, for cache keying.

    Includes the CoACD version, because its splits change between releases, so a
    cache written by another version is not the geometry this install builds.
    Both packages of the ``decomp`` extra must be present: without
    ``threadpoolctl`` the search is not thread-pinned, and ``_convex_parts``
    falls back to a single hull rather than write geometry it cannot reproduce.
    """
    try:
        from importlib.metadata import PackageNotFoundError, version

        version("threadpoolctl")
        return f"coacd {version('coacd')}"
    except (ImportError, PackageNotFoundError):
        return _HULL_DECOMPOSER


def _convex_parts(mesh: _ExportableMesh) -> list[_ExportableMesh]:
    """Decompose ``mesh`` into convex collision parts.

    Nearly convex models keep their single convex hull. The rest are handed to
    CoACD, which is an optional extra (``pip install so101-nexus[decomp]``);
    without it the single hull remains the fallback, so no existing install
    breaks, it just keeps the coarse collision geometry.

    CoACD's MCTS search is parallelized over OpenMP, and its thread scheduling
    changes the split it converges on: the same mesh and seed yield different
    part counts run to run. Scene geometry must not vary between runs, so the
    search is pinned to one thread (a few seconds per model, once, then cached).
    The limit is scoped to this call and restored afterwards.
    """
    hull = mesh.convex_hull
    hull_volume = abs(hull.volume)
    # The YCB scans are open surfaces, so this is the divergence-theorem volume
    # rather than a true enclosed one; logged because a model whose integral
    # overshoots would skip decomposition without any other signal.
    convexity = abs(mesh.volume) / hull_volume if hull_volume > 0.0 else 1.0
    logger.debug("mesh convexity (volume / hull volume): %.3f", convexity)
    if convexity >= _SINGLE_HULL_CONVEXITY:
        return [hull]
    try:
        import coacd
        from threadpoolctl import threadpool_limits
    except ImportError as exc:
        logger.warning(
            "%s is not installed: collision geometry stays a single convex hull, "
            "which physics sees as a solid wedge for concave models. "
            "Install so101-nexus[decomp] for a multi-hull decomposition.",
            exc.name or "coacd",
        )
        return [hull]

    import trimesh

    coacd.set_log_level("error")
    with threadpool_limits(limits=1, user_api="openmp"):
        parts = coacd.run_coacd(
            coacd.Mesh(mesh.vertices, mesh.faces),
            threshold=_COACD_THRESHOLD,
            max_convex_hull=_COACD_MAX_HULLS,
            preprocess_mode="on",
            preprocess_resolution=_COACD_PREPROCESS_RESOLUTION,
            seed=_COACD_SEED,
        )
    # CoACD parts are near-convex, not exactly convex; MuJoCo collides the hull
    # of a mesh geom anyway, so hulling here keeps the exported OBJ and the
    # compiled collider identical. A sliver part can fail to hull at all, which
    # is the same "drop it" case as a zero-volume one.
    hulls = []
    for vertices, faces in parts:
        try:
            hulls.append(trimesh.Trimesh(vertices, faces).convex_hull)
        except Exception:  # qhull errors are not importable without scipy here
            logger.debug("dropping a degenerate CoACD part")
    return [part for part in hulls if abs(part.volume) > 0.0] or [hull]


def _write_collision_parts(mesh: _ExportableMesh, out_dir: Path) -> None:
    """Export the convex decomposition of ``mesh`` and its manifest into ``out_dir``.

    The manifest is the cache sentinel, so it is removed before the parts are
    rewritten and published atomically: an interrupted export leaves no manifest
    and the next call rebuilds, instead of wedging the cache behind a truncated
    one that ``ensure_ycb_assets`` would accept forever.
    """
    parts = _convex_parts(mesh)
    volumes = [abs(part.volume) for part in parts]
    total = sum(volumes)
    fractions = (
        [volume / total for volume in volumes] if total > 0.0 else [1.0 / len(parts)] * len(parts)
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / _MANIFEST_NAME
    manifest_path.unlink(missing_ok=True)
    for stale in out_dir.glob("collision_*.obj"):
        stale.unlink()
    entries = []
    for index, (part, fraction) in enumerate(zip(parts, fractions, strict=True)):
        name = f"collision_{index:03d}.obj"
        part.export(str(out_dir / name), file_type="obj")
        entries.append({"file": name, "mass_fraction": fraction})
    pending = manifest_path.with_suffix(".json.pending")
    pending.write_text(
        json.dumps({"decomposer": _decomposer_id(), "parts": entries}, indent=2),
        encoding="utf-8",
    )
    os.replace(pending, manifest_path)


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


def _read_manifest(collision_dir: Path) -> dict | None:
    """Return the parsed parts manifest, or ``None`` when it is absent or unusable.

    Anything unreadable reads as absent so the cache rebuilds itself: the
    manifest is the sentinel ``ensure_ycb_assets`` early-returns on, and a
    truncated one would otherwise wedge the model behind a parse error forever.
    """
    try:
        manifest = json.loads((collision_dir / _MANIFEST_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(manifest, dict) or not isinstance(manifest.get("parts"), list):
        return None
    return manifest


def _collision_parts_are_current(collision_dir: Path) -> bool:
    """Return True when the cached parts were built by this install's decomposer.

    A cache written by a different CoACD version (or by the single-hull fallback
    before the ``decomp`` extra was installed) is rebuilt, so scene geometry
    always matches the decomposer that is actually present. The reverse is never
    true: an install without CoACD keeps whatever is cached rather than
    overwriting a real decomposition with a coarse hull.
    """
    manifest = _read_manifest(collision_dir)
    if manifest is None:
        return False
    current = _decomposer_id()
    return manifest.get("decomposer") == current or current == _HULL_DECOMPOSER


def ensure_ycb_assets(model_id: str) -> Path:
    """Download YCB mesh assets from HuggingFace if not already cached.

    Downloads from the ai-habitat/ycb dataset and converts GLB meshes to OBJ
    format for use with MuJoCo. The visual mesh is exported directly; the
    collision geometry is a cached convex decomposition of it (see
    :func:`get_ycb_collision_parts`), so physics sees the concavities a single
    hull would fill in. Decomposition needs the optional ``decomp`` extra;
    without it the collision geometry falls back to one convex hull, which is
    what ``YCBObject`` warns about.

    Returns the directory containing the model's mesh files.
    """
    _validate_model_id(model_id)
    mesh_dir = _CACHE_DIR / model_id

    collision_dir = _collision_dir(model_id)
    visual_path = mesh_dir / "visual.obj"
    texture_path = mesh_dir / "texture.png"
    glb_path = _CACHE_DIR / "meshes" / model_id / "google_16k" / "textured.glb"
    orig_path = glb_path.with_suffix(".glb.orig")

    if _collision_parts_are_current(collision_dir) and visual_path.exists():
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

    _write_collision_parts(_load_exportable_mesh(glb_path), collision_dir)
    # Single-hull cache from a release before the decomposition; nothing reads it.
    (mesh_dir / "collision.obj").unlink(missing_ok=True)
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


def get_ycb_collision_parts(model_id: str) -> list[YCBCollisionPart]:
    """Return the cached convex collision parts of a YCB model, in export order.

    Raises
    ------
    FileNotFoundError
        If the model has not been prepared yet; call
        :func:`ensure_ycb_assets` first.
    """
    _validate_model_id(model_id)
    manifest = _read_manifest(_collision_dir(model_id))
    if manifest is None:
        raise FileNotFoundError(
            f"No usable collision decomposition cached for {model_id!r}; "
            f"call ensure_ycb_assets({model_id!r}) first."
        )
    parts_dir = _collision_dir(model_id)
    return [
        YCBCollisionPart(parts_dir / entry["file"], float(entry["mass_fraction"]))
        for entry in manifest["parts"]
    ]


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
