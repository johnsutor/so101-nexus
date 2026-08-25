"""YCB mesh asset management - downloads from HuggingFace on demand."""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import math
import os
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypedDict, cast

from so101_nexus.constants import YCB_OBJECTS

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)

_HF_REPO_ID = os.environ.get("SO101_YCB_HF_REPO", "ai-habitat/ycb")
_CACHE_DIR = Path.home() / ".cache" / "so101_nexus" / "ycb"

_COLLISION_SUBDIR = "collision_v3"
"""Cache subdirectory for collision parts built with a measured surface-error gate."""

_MANIFEST_NAME = "manifest.json"

_FALLBACK_DECOMPOSER = "convex_hull"
"""Manifest marker for a hull built without the ``decomp`` extra."""

_HULL_DECOMPOSER = "none (hull kept)"
"""Manifest marker for a measured hull that does not need decomposition."""

_HULL_GAP_SAMPLES = 20_000
_HULL_GAP_SEED = 0
_HULL_GAP_GATE_M = 0.003
_HULL_GAP_MAX_GATE_M = 0.010


class _CoacdSettings(TypedDict):
    threshold: float
    max_convex_hull: int
    preprocess_mode: str
    preprocess_resolution: int
    resolution: int
    mcts_nodes: int
    mcts_iterations: int
    mcts_max_depth: int
    pca: bool
    merge: bool
    decimate: bool
    max_ch_vertex: int
    seed: int


_COACD_SETTINGS: _CoacdSettings = {
    "threshold": 0.03,
    "max_convex_hull": -1,
    "preprocess_mode": "auto",
    "preprocess_resolution": 50,
    "resolution": 2000,
    "mcts_nodes": 20,
    "mcts_iterations": 150,
    "mcts_max_depth": 3,
    "pca": False,
    "merge": True,
    "decimate": True,
    "max_ch_vertex": 128,
    "seed": 0,
}
"""Audited CoACD configuration for the ten supported YCB scans."""


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


@dataclass(frozen=True, slots=True)
class _CollisionBuild:
    """Collision parts and the inputs that produced them."""

    parts: tuple[_ExportableMesh, ...]
    decomposer: str
    decomposed: bool
    settings: Mapping[str, object] | None
    hull_gap_p95_m: float | None
    hull_gap_max_m: float | None


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


def _load_exportable_mesh(mesh_path: Path) -> _ExportableMesh:
    """Load a mesh file as one object with ``export`` and ``convex_hull``."""
    import trimesh

    scene_or_mesh = trimesh.load(str(mesh_path), force="mesh")
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
    """Identify the installed generator for cache keying."""
    try:
        from importlib.metadata import PackageNotFoundError, version

        for package in ("threadpoolctl", "rtree"):
            version(package)
        return f"coacd {version('coacd')}"
    except (ImportError, PackageNotFoundError):
        return _FALLBACK_DECOMPOSER


def _hull_surface_gap(
    hull: _ExportableMesh,
    mesh: _ExportableMesh,
) -> tuple[float, float]:
    """Return the p95 and maximum distances from a hull surface to a scan."""
    import numpy as np
    import trimesh

    points, _ = trimesh.sample.sample_surface(
        hull,
        _HULL_GAP_SAMPLES,
        seed=_HULL_GAP_SEED,
    )
    scan = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
    scan.update_faces(scan.nondegenerate_faces())
    distances = np.asarray(trimesh.proximity.closest_point(scan, points)[1], dtype=np.float64)
    if distances.size == 0 or not np.isfinite(distances).all():
        raise ValueError("The hull surface distance contains no finite samples.")
    return float(np.percentile(distances, 95)), float(distances.max())


def _build_collision_geometry(mesh: _ExportableMesh) -> _CollisionBuild:
    """Build a measured hull or a collision-aware convex decomposition."""
    hull = mesh.convex_hull
    try:
        import coacd
        import trimesh
        from threadpoolctl import threadpool_limits

        importlib.import_module("rtree")
    except ImportError as exc:
        logger.warning(
            "%s is not installed: collision geometry stays a single convex hull. "
            "Install so101-nexus[decomp] for measured multi-hull geometry.",
            exc.name or "a decomposition dependency",
        )
        return _CollisionBuild(
            parts=(hull,),
            decomposer=_FALLBACK_DECOMPOSER,
            decomposed=False,
            settings=None,
            hull_gap_p95_m=None,
            hull_gap_max_m=None,
        )

    hull_gap_p95_m, hull_gap_max_m = _hull_surface_gap(hull, mesh)
    logger.debug(
        "hull surface gap: p95=%.6f m, max=%.6f m",
        hull_gap_p95_m,
        hull_gap_max_m,
    )
    if hull_gap_p95_m <= _HULL_GAP_GATE_M and hull_gap_max_m <= _HULL_GAP_MAX_GATE_M:
        return _CollisionBuild(
            parts=(hull,),
            decomposer=_HULL_DECOMPOSER,
            decomposed=False,
            settings=None,
            hull_gap_p95_m=hull_gap_p95_m,
            hull_gap_max_m=hull_gap_max_m,
        )

    coacd.set_log_level("error")
    with threadpool_limits(limits=1, user_api="openmp"):
        pieces = coacd.run_coacd(
            coacd.Mesh(mesh.vertices, mesh.faces),
            **_COACD_SETTINGS,
        )
    parts = tuple(
        cast("_ExportableMesh", trimesh.Trimesh(vertices, faces)) for vertices, faces in pieces
    )
    parts = tuple(part for part in parts if abs(part.volume) > 0.0)
    if not parts:
        raise RuntimeError("CoACD produced no positive-volume collision parts.")
    max_vertices = int(_COACD_SETTINGS["max_ch_vertex"])
    if any(len(part.vertices) > max_vertices for part in parts):
        raise RuntimeError(
            f"CoACD produced a collision part with more than {max_vertices} vertices."
        )
    return _CollisionBuild(
        parts=parts,
        decomposer=_decomposer_id(),
        decomposed=True,
        settings=_COACD_SETTINGS,
        hull_gap_p95_m=hull_gap_p95_m,
        hull_gap_max_m=hull_gap_max_m,
    )


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_collision_parts(
    mesh: _ExportableMesh,
    out_dir: Path,
    *,
    model_id: str,
    source_path: Path,
) -> None:
    """Export collision parts and their generation manifest into ``out_dir``."""
    build = _build_collision_geometry(mesh)
    volumes = [abs(part.volume) for part in build.parts]
    total = sum(volumes)
    fractions = [volume / total for volume in volumes]

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / _MANIFEST_NAME
    manifest_path.unlink(missing_ok=True)
    for stale in out_dir.glob("collision_*.obj"):
        stale.unlink()
    entries = []
    for index, (part, volume, fraction) in enumerate(
        zip(build.parts, volumes, fractions, strict=True)
    ):
        name = f"collision_{index:03d}.obj"
        part.export(str(out_dir / name), file_type="obj")
        entries.append(
            {
                "file": name,
                "mass_fraction": fraction,
                "volume_m3": volume,
                "n_vertices": len(part.vertices),
            }
        )
    manifest = {
        "model_id": model_id,
        "source": source_path.name,
        "source_sha256": _sha256(source_path),
        "decomposer": build.decomposer,
        "settings": dict(build.settings) if build.settings is not None else None,
        "hull_gap_p95_m": build.hull_gap_p95_m,
        "hull_gap_max_m": build.hull_gap_max_m,
        "hull_gap_gate_m": _HULL_GAP_GATE_M,
        "hull_gap_max_gate_m": _HULL_GAP_MAX_GATE_M,
        "decomposed": build.decomposed,
        "mass_split": "part volume / total part volume",
        "parts": entries,
    }
    pending = manifest_path.with_suffix(".json.pending")
    pending.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
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


def _manifest_part_is_usable(part: object) -> bool:
    if not isinstance(part, Mapping):
        return False
    part = cast("Mapping[str, object]", part)
    name = part.get("file")
    fraction_value = part.get("mass_fraction")
    if not isinstance(name, str) or Path(name).name != name:
        return False
    if isinstance(fraction_value, bool) or not isinstance(fraction_value, (int, float)):
        return False
    fraction = float(fraction_value)
    return math.isfinite(fraction) and fraction >= 0.0


def _read_manifest(collision_dir: Path) -> dict | None:
    """Return a usable collision manifest, or ``None``."""
    try:
        manifest = json.loads((collision_dir / _MANIFEST_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(manifest, dict) or not isinstance(manifest.get("parts"), list):
        return None
    parts = manifest["parts"]
    if not parts or not all(_manifest_part_is_usable(part) for part in parts):
        return None
    return manifest


def _manifest_source_matches(collision_dir: Path, manifest: Mapping[str, object]) -> bool:
    source = collision_dir.parent / str(manifest.get("source", ""))
    try:
        return manifest.get("source_sha256") == _sha256(source)
    except OSError:
        return False


def _manifest_generator_matches(manifest: Mapping[str, object]) -> bool:
    current = _decomposer_id()
    if current == _FALLBACK_DECOMPOSER:
        return True
    gates_match = (
        manifest.get("hull_gap_gate_m") == _HULL_GAP_GATE_M
        and manifest.get("hull_gap_max_gate_m") == _HULL_GAP_MAX_GATE_M
    )
    if not gates_match:
        return False
    if manifest.get("decomposed"):
        return manifest.get("decomposer") == current and manifest.get("settings") == _COACD_SETTINGS
    return (
        manifest.get("decomposer") == _HULL_DECOMPOSER
        and manifest.get("settings") is None
        and manifest.get("hull_gap_p95_m") is not None
        and manifest.get("hull_gap_max_m") is not None
    )


def _collision_parts_are_current(collision_dir: Path) -> bool:
    """Return true when all cached parts match the scan and generator inputs."""
    manifest = _read_manifest(collision_dir)
    if manifest is None:
        return False
    parts_exist = all((collision_dir / part["file"]).is_file() for part in manifest["parts"])
    return (
        parts_exist
        and _manifest_source_matches(collision_dir, manifest)
        and _manifest_generator_matches(manifest)
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

    _write_collision_parts(
        _load_exportable_mesh(visual_path),
        collision_dir,
        model_id=model_id,
        source_path=visual_path,
    )
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
