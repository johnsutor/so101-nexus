from __future__ import annotations

import contextlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from so101_nexus import ycb_assets
from so101_nexus.constants import YCB_OBJECTS
from so101_nexus.ycb_assets import get_ycb_mesh_dir

EXPECTED_MODEL_IDS = [
    "009_gelatin_box",
    "011_banana",
    "030_fork",
    "031_spoon",
    "032_knife",
    "033_spatula",
    "037_scissors",
    "040_large_marker",
    "043_phillips_screwdriver",
    "058_golf_ball",
]


class TestYCBConstants:
    def test_ycb_objects_values_are_strings(self):
        for model_id, name in YCB_OBJECTS.items():
            assert isinstance(name, str), f"{model_id} name is not a string"
            assert len(name) > 0, f"{model_id} name is empty"


class TestYCBAssets:
    def test_get_ycb_mesh_dir_returns_path(self):
        result = get_ycb_mesh_dir("009_gelatin_box")
        assert isinstance(result, Path)

    def test_get_ycb_mesh_dir_invalid_model_raises(self):
        with pytest.raises(ValueError, match="model_id"):
            get_ycb_mesh_dir("invalid_model")

    def test_get_ycb_mesh_dir_contains_model_id(self):
        for model_id in EXPECTED_MODEL_IDS:
            path = get_ycb_mesh_dir(model_id)
            assert model_id in str(path)


class _FakeMesh:
    """Stands in for a trimesh mesh; convex by construction, so no decomposition."""

    volume = 1.0
    vertices = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    faces = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))

    def __init__(self):
        self.exports: list[tuple[str, str | None]] = []
        self.convex_hull = self

    def export(self, file_obj: str, file_type: str | None = None):
        self.exports.append((file_obj, file_type))
        Path(file_obj).write_text("hull", encoding="utf-8")


class _FakeScene:
    def __init__(self, mesh: _FakeMesh):
        self._mesh = mesh

    def dump(self, concatenate: bool = False):
        assert concatenate
        return self._mesh


class _FakeImage:
    def __init__(self):
        self.saved: list[tuple[str, str | None]] = []

    def save(self, path: str, format: str | None = None):
        self.saved.append((path, format))
        Path(path).write_text("texture", encoding="utf-8")


class _FakeMaterial:
    def __init__(self, image: object | None = None, base_color_texture: object | None = None):
        self.image = image
        self.baseColorTexture = base_color_texture


class _FakeVisual:
    def __init__(self, material: _FakeMaterial):
        self.material = material


class _FakeTexturedMesh(_FakeMesh):
    def __init__(self, image: object | None = None, base_color_texture: object | None = None):
        super().__init__()
        self.visual = _FakeVisual(_FakeMaterial(image, base_color_texture))


def _patch_module(monkeypatch: pytest.MonkeyPatch, name: str, module: object) -> None:
    monkeypatch.setitem(sys.modules, name, module)


def test_convert_glb_to_obj_handles_scene(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    mesh = _FakeMesh()
    fake_trimesh = types.SimpleNamespace(
        Scene=_FakeScene,
        load=lambda *_args, **_kwargs: _FakeScene(mesh),
    )
    _patch_module(monkeypatch, "trimesh", fake_trimesh)

    out = tmp_path / "visual.obj"
    ycb_assets._convert_glb_to_obj(tmp_path / "in.glb", out)
    assert mesh.exports == [(str(out), "obj")]


def test_convert_glb_to_obj_handles_mesh(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    mesh = _FakeMesh()
    fake_trimesh = types.SimpleNamespace(
        Scene=_FakeScene,
        load=lambda *_args, **_kwargs: mesh,
    )
    _patch_module(monkeypatch, "trimesh", fake_trimesh)

    out = tmp_path / "visual.obj"
    ycb_assets._convert_glb_to_obj(tmp_path / "in.glb", out)
    assert mesh.exports == [(str(out), "obj")]


def test_get_ycb_texture_file_returns_cache_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(ycb_assets, "_CACHE_DIR", tmp_path)

    assert (
        ycb_assets.get_ycb_texture_file("058_golf_ball")
        == tmp_path / "058_golf_ball" / "texture.png"
    )


def test_extract_glb_texture_saves_material_image(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    image = _FakeImage()
    mesh = _FakeTexturedMesh(image=image)
    fake_trimesh = types.SimpleNamespace(
        Scene=_FakeScene,
        load=lambda *_args, **_kwargs: mesh,
    )
    _patch_module(monkeypatch, "trimesh", fake_trimesh)

    out = tmp_path / "texture.png"

    assert ycb_assets._extract_glb_texture(tmp_path / "textured.glb", out) is True
    assert image.saved == [(str(out), "PNG")]
    assert out.read_text(encoding="utf-8") == "texture"


def test_extract_glb_texture_returns_false_without_texture(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    mesh = _FakeTexturedMesh()
    fake_trimesh = types.SimpleNamespace(
        Scene=_FakeScene,
        load=lambda *_args, **_kwargs: mesh,
    )
    _patch_module(monkeypatch, "trimesh", fake_trimesh)

    out = tmp_path / "texture.png"

    assert ycb_assets._extract_glb_texture(tmp_path / "textured.glb", out) is False
    assert not out.exists()


def _write_cached_parts(
    mesh_dir: Path,
    files: tuple[str, ...] = ("collision_000.obj",),
    decomposer: str | None = None,
) -> Path:
    """Write a collision-v3 cache entry and return its directory."""
    mesh_dir.mkdir(parents=True, exist_ok=True)
    source = mesh_dir / "visual.obj"
    source.write_text("v", encoding="utf-8")
    parts_dir = mesh_dir / "collision_v3"
    parts_dir.mkdir(parents=True, exist_ok=True)
    for name in files:
        (parts_dir / name).write_text("c", encoding="utf-8")
    decomposed = decomposer is not None
    manifest = {
        "model_id": mesh_dir.name,
        "source": source.name,
        "source_sha256": ycb_assets._sha256(source),
        "decomposer": decomposer or ycb_assets._HULL_DECOMPOSER,
        "settings": ycb_assets._COACD_SETTINGS if decomposed else None,
        "hull_gap_p95_m": 0.0,
        "hull_gap_max_m": 0.0,
        "hull_gap_gate_m": ycb_assets._HULL_GAP_GATE_M,
        "hull_gap_max_gate_m": ycb_assets._HULL_GAP_MAX_GATE_M,
        "decomposed": decomposed,
        "parts": [{"file": name, "mass_fraction": 1.0 / len(files)} for name in files],
    }
    (parts_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return parts_dir


def test_ensure_ycb_assets_cache_hit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(ycb_assets, "_CACHE_DIR", tmp_path)
    model_id = "009_gelatin_box"
    mesh_dir = tmp_path / model_id
    mesh_dir.mkdir(parents=True)
    _write_cached_parts(mesh_dir)
    (mesh_dir / "visual.obj").write_text("v", encoding="utf-8")
    (mesh_dir / "texture.png").write_text("t", encoding="utf-8")

    def _unexpected_snapshot_download(**_kwargs):
        raise AssertionError("snapshot_download should not be called on cache hit")

    _patch_module(
        monkeypatch,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=_unexpected_snapshot_download),
    )

    result = ycb_assets.ensure_ycb_assets(model_id)
    assert result == mesh_dir


def test_ensure_ycb_assets_cache_hit_extracts_missing_texture(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setattr(ycb_assets, "_CACHE_DIR", tmp_path)
    model_id = "009_gelatin_box"
    mesh_dir = tmp_path / model_id
    mesh_dir.mkdir(parents=True)
    _write_cached_parts(mesh_dir)
    (mesh_dir / "visual.obj").write_text("v", encoding="utf-8")
    glb = tmp_path / "meshes" / model_id / "google_16k" / "textured.glb"
    glb.parent.mkdir(parents=True)
    glb.write_text("fake-glb", encoding="utf-8")
    glb_orig = glb.with_suffix(".glb.orig")
    glb_orig.write_text("fake-glb-orig", encoding="utf-8")
    calls: list[tuple[Path, Path]] = []

    def _extract(glb_path: Path, texture_path: Path) -> bool:
        calls.append((glb_path, texture_path))
        texture_path.write_text("texture", encoding="utf-8")
        return True

    monkeypatch.setattr(ycb_assets, "_extract_glb_texture", _extract)

    result = ycb_assets.ensure_ycb_assets(model_id)

    assert result == mesh_dir
    assert calls == [(glb_orig, mesh_dir / "texture.png")]


def test_ensure_ycb_assets_returns_when_texture_extract_finds_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    model_id = "009_gelatin_box"
    monkeypatch.setattr(ycb_assets, "_CACHE_DIR", tmp_path)

    def _snapshot_download(**_kwargs):
        glb = tmp_path / "meshes" / model_id / "google_16k"
        glb.mkdir(parents=True, exist_ok=True)
        (glb / "textured.glb").write_text("fake-glb", encoding="utf-8")

    _patch_module(
        monkeypatch,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=_snapshot_download),
    )
    monkeypatch.setattr(ycb_assets, "_convert_glb_to_obj", lambda _g, p: p.write_text("v"))
    monkeypatch.setattr(ycb_assets, "_extract_glb_texture", lambda _g, _p: False)
    fake_trimesh = types.SimpleNamespace(Scene=_FakeScene, load=lambda *_a, **_k: _FakeMesh())
    _patch_module(monkeypatch, "trimesh", fake_trimesh)
    _fake_coacd(monkeypatch, [])
    monkeypatch.setattr(ycb_assets, "_hull_surface_gap", lambda _h, _m: (0.0, 0.0))

    mesh_dir = ycb_assets.ensure_ycb_assets(model_id)

    assert mesh_dir == tmp_path / model_id
    assert not (mesh_dir / "texture.png").exists()


def test_ensure_ycb_assets_download_and_decomposition(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    model_id = "009_gelatin_box"
    monkeypatch.setattr(ycb_assets, "_CACHE_DIR", tmp_path)

    called: dict[str, object] = {}
    mesh = _FakeMesh()

    def _snapshot_download(**kwargs):
        called["snapshot_kwargs"] = kwargs
        glb = tmp_path / "meshes" / model_id / "google_16k"
        glb.mkdir(parents=True, exist_ok=True)
        (glb / "textured.glb").write_text("fake-glb", encoding="utf-8")

    def _convert_glb_to_obj(glb_path: Path, obj_path: Path):
        called["convert_args"] = (glb_path, obj_path)
        obj_path.write_text("visual", encoding="utf-8")

    fake_trimesh = types.SimpleNamespace(Scene=_FakeScene, load=lambda *_a, **_k: mesh)
    _patch_module(monkeypatch, "trimesh", fake_trimesh)
    _patch_module(
        monkeypatch,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=_snapshot_download),
    )
    monkeypatch.setattr(ycb_assets, "_convert_glb_to_obj", _convert_glb_to_obj)
    _fake_coacd(monkeypatch, [])
    monkeypatch.setattr(ycb_assets, "_hull_surface_gap", lambda _h, _m: (0.0, 0.0))

    mesh_dir = ycb_assets.ensure_ycb_assets(model_id)
    assert mesh_dir == tmp_path / model_id
    assert "snapshot_kwargs" in called
    assert called["convert_args"] == (
        tmp_path / "meshes" / model_id / "google_16k" / "textured.glb",
        tmp_path / model_id / "visual.obj",
    )
    parts_dir = tmp_path / model_id / "collision_v3"
    assert mesh.exports[-1] == (str(parts_dir / "collision_000.obj"), "obj")
    manifest = json.loads((parts_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source"] == "visual.obj"
    assert manifest["source_sha256"] == ycb_assets._sha256(mesh_dir / "visual.obj")
    assert manifest["parts"] == [
        {
            "file": "collision_000.obj",
            "mass_fraction": 1.0,
            "volume_m3": 1.0,
            "n_vertices": 4,
        }
    ]


def test_ensure_ycb_assets_scene_path_for_hull(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    model_id = "009_gelatin_box"
    monkeypatch.setattr(ycb_assets, "_CACHE_DIR", tmp_path)

    mesh = _FakeMesh()
    fake_scene = _FakeScene(mesh)

    def _snapshot_download(**_kwargs):
        glb = tmp_path / "meshes" / model_id / "google_16k"
        glb.mkdir(parents=True, exist_ok=True)
        (glb / "textured.glb").write_text("fake-glb", encoding="utf-8")

    fake_trimesh = types.SimpleNamespace(Scene=_FakeScene, load=lambda *_a, **_k: fake_scene)
    _patch_module(monkeypatch, "trimesh", fake_trimesh)
    _patch_module(
        monkeypatch,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=_snapshot_download),
    )
    monkeypatch.setattr(ycb_assets, "_convert_glb_to_obj", lambda _g, p: p.write_text("v"))
    _fake_coacd(monkeypatch, [])
    monkeypatch.setattr(ycb_assets, "_hull_surface_gap", lambda _h, _m: (0.0, 0.0))

    ycb_assets.ensure_ycb_assets(model_id)
    assert mesh.exports[-1] == (
        str(tmp_path / model_id / "collision_v3" / "collision_000.obj"),
        "obj",
    )


def test_collision_and_visual_mesh_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(ycb_assets, "_CACHE_DIR", tmp_path)
    model_id = "058_golf_ball"
    parts_dir = _write_cached_parts(tmp_path / model_id, ("collision_000.obj", "collision_001.obj"))
    assert ycb_assets.get_ycb_collision_meshes(model_id) == [
        parts_dir / "collision_000.obj",
        parts_dir / "collision_001.obj",
    ]
    assert ycb_assets.get_ycb_collision_mesh(model_id) == parts_dir / "collision_000.obj"
    assert ycb_assets.get_ycb_visual_mesh(model_id) == tmp_path / model_id / "visual.obj"


def test_collision_parts_require_prepared_assets(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(ycb_assets, "_CACHE_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="ensure_ycb_assets"):
        ycb_assets.get_ycb_collision_parts("058_golf_ball")


def test_collision_parts_reject_a_truncated_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """A half-written manifest must read as absent, not wedge the cache forever."""
    monkeypatch.setattr(ycb_assets, "_CACHE_DIR", tmp_path)
    model_id = "058_golf_ball"
    parts_dir = _write_cached_parts(tmp_path / model_id)
    (parts_dir / "manifest.json").write_text('{"parts": [{"file":', encoding="utf-8")

    assert not ycb_assets._collision_parts_are_current(parts_dir)
    with pytest.raises(FileNotFoundError, match="ensure_ycb_assets"):
        ycb_assets.get_ycb_collision_parts(model_id)


def test_cached_parts_from_another_decomposer_are_rebuilt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """Installing the extra (or upgrading CoACD) must not keep the old geometry.

    The reverse never holds: an install without CoACD keeps a real decomposition
    rather than overwriting it with a coarse hull.
    """
    monkeypatch.setattr(ycb_assets, "_CACHE_DIR", tmp_path)
    parts_dir = _write_cached_parts(tmp_path / "058_golf_ball", decomposer="coacd 0.0.1")
    monkeypatch.setattr(ycb_assets, "_decomposer_id", lambda: "coacd 9.9.9")
    assert not ycb_assets._collision_parts_are_current(parts_dir)

    monkeypatch.setattr(ycb_assets, "_decomposer_id", lambda: ycb_assets._FALLBACK_DECOMPOSER)
    assert ycb_assets._collision_parts_are_current(parts_dir)


def test_cached_parts_are_rebuilt_when_the_visual_scan_changes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """The source hash must prevent parts from another scan from reaching physics."""
    monkeypatch.setattr(ycb_assets, "_CACHE_DIR", tmp_path)
    mesh_dir = tmp_path / "058_golf_ball"
    parts_dir = _write_cached_parts(mesh_dir)
    assert ycb_assets._collision_parts_are_current(parts_dir)

    (mesh_dir / "visual.obj").write_text("changed", encoding="utf-8")

    assert not ycb_assets._collision_parts_are_current(parts_dir)


def test_extract_glb_texture_accepts_orig_extension(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """trimesh.load must receive file_type='glb' so .glb.orig paths load."""
    image = _FakeImage()
    mesh = _FakeTexturedMesh(image=image)
    load_calls: list[tuple[str, dict]] = []

    def _load(path: str, **kwargs):
        load_calls.append((path, dict(kwargs)))
        return mesh

    fake_trimesh = types.SimpleNamespace(Scene=_FakeScene, load=_load)
    _patch_module(monkeypatch, "trimesh", fake_trimesh)

    out = tmp_path / "texture.png"
    orig_path = tmp_path / "textured.glb.orig"

    assert ycb_assets._extract_glb_texture(orig_path, out) is True
    assert load_calls, "trimesh.load was not invoked"
    _path, kwargs = load_calls[0]
    assert kwargs.get("file_type") == "glb", (
        "trimesh.load must be called with file_type='glb' so .orig is accepted; "
        f"got kwargs={kwargs}"
    )


def test_texture_glb_path_prefers_orig(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """When both textured.glb.orig and textured.glb exist, the .orig wins."""
    monkeypatch.setattr(ycb_assets, "_CACHE_DIR", tmp_path)
    model_id = "011_banana"
    base = tmp_path / "meshes" / model_id / "google_16k"
    base.mkdir(parents=True)
    glb = base / "textured.glb"
    orig = base / "textured.glb.orig"
    glb.write_text("stripped", encoding="utf-8")
    orig.write_text("orig-with-texture", encoding="utf-8")

    assert ycb_assets._texture_glb_path(model_id) == orig


def test_texture_glb_path_falls_back_to_glb(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """When only textured.glb exists, _texture_glb_path returns it."""
    monkeypatch.setattr(ycb_assets, "_CACHE_DIR", tmp_path)
    model_id = "011_banana"
    base = tmp_path / "meshes" / model_id / "google_16k"
    base.mkdir(parents=True)
    glb = base / "textured.glb"
    glb.write_text("stripped", encoding="utf-8")

    assert ycb_assets._texture_glb_path(model_id) == glb


def test_ensure_ycb_assets_downloads_orig_when_partial_cache_lacks_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """Partial caches with only textured.glb must pull .orig before extraction."""
    monkeypatch.setattr(ycb_assets, "_CACHE_DIR", tmp_path)
    model_id = "011_banana"

    mesh_dir = tmp_path / model_id
    mesh_dir.mkdir(parents=True)
    _write_cached_parts(mesh_dir)
    (mesh_dir / "visual.obj").write_text("v", encoding="utf-8")
    glb_dir = tmp_path / "meshes" / model_id / "google_16k"
    glb_dir.mkdir(parents=True)
    (glb_dir / "textured.glb").write_text("stripped", encoding="utf-8")

    download_called = {"count": 0}

    def _snapshot_download(**_kwargs):
        download_called["count"] += 1
        (glb_dir / "textured.glb.orig").write_text("orig", encoding="utf-8")

    _patch_module(
        monkeypatch,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=_snapshot_download),
    )

    extract_calls: list[Path] = []

    def _extract(glb_path: Path, texture_path: Path) -> bool:
        extract_calls.append(glb_path)
        texture_path.write_text("png-bytes", encoding="utf-8")
        return True

    monkeypatch.setattr(ycb_assets, "_extract_glb_texture", _extract)

    result = ycb_assets.ensure_ycb_assets(model_id)
    assert result == mesh_dir
    assert download_called["count"] == 1, (
        "Expected snapshot_download to be triggered to pull textured.glb.orig"
    )
    assert extract_calls == [glb_dir / "textured.glb.orig"], (
        f"Texture must be extracted from .orig after the download, got {extract_calls}"
    )
    assert (mesh_dir / "texture.png").exists()


def test_ensure_ycb_assets_logs_warning_when_extraction_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    """Failed texture extraction must warn with the model and path tried."""
    import logging

    monkeypatch.setattr(ycb_assets, "_CACHE_DIR", tmp_path)
    model_id = "011_banana"

    mesh_dir = tmp_path / model_id
    mesh_dir.mkdir(parents=True)
    _write_cached_parts(mesh_dir)
    (mesh_dir / "visual.obj").write_text("v", encoding="utf-8")
    glb_dir = tmp_path / "meshes" / model_id / "google_16k"
    glb_dir.mkdir(parents=True)
    (glb_dir / "textured.glb").write_text("stripped", encoding="utf-8")
    (glb_dir / "textured.glb.orig").write_text("orig", encoding="utf-8")

    monkeypatch.setattr(ycb_assets, "_extract_glb_texture", lambda _g, _t: False)

    with caplog.at_level(logging.WARNING, logger="so101_nexus.ycb_assets"):
        result = ycb_assets.ensure_ycb_assets(model_id)

    assert result == mesh_dir
    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warning_records) == 1, (
        f"Expected exactly one WARNING; got {len(warning_records)}: "
        f"{[r.getMessage() for r in warning_records]}"
    )
    msg = warning_records[0].getMessage()
    assert model_id in msg, f"WARNING must name the model id, got: {msg!r}"
    assert "textured.glb.orig" in msg, f"WARNING must name the GLB path tried, got: {msg!r}"


class _ConcaveMesh:
    """Mesh whose convex hull is twice its volume, so it must be decomposed."""

    volume = 0.5
    vertices = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    faces = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))

    def __init__(self):
        self.convex_hull = _FakeMesh()


def _fake_coacd(monkeypatch: pytest.MonkeyPatch, events: list[tuple]) -> None:
    """Install fake ``coacd`` and ``threadpoolctl`` modules recording their use.

    Both live in the optional ``decomp`` extra, so faking them keeps these tests
    running on a plain install (and in CI, which syncs no extras).
    """

    def _run_coacd(mesh, **kwargs):
        events.append(("run_coacd", kwargs))
        return [("v0", "f0"), ("v1", "f1")]

    @contextlib.contextmanager
    def _threadpool_limits(limits=None, user_api=None):
        events.append(("enter_limits", limits, user_api))
        yield
        events.append(("exit_limits",))

    _patch_module(
        monkeypatch,
        "coacd",
        types.SimpleNamespace(
            Mesh=lambda vertices, faces: (vertices, faces),
            run_coacd=_run_coacd,
            set_log_level=lambda _level: None,
        ),
    )
    _patch_module(
        monkeypatch, "threadpoolctl", types.SimpleNamespace(threadpool_limits=_threadpool_limits)
    )
    _patch_module(monkeypatch, "rtree", types.SimpleNamespace())


def _fake_trimesh_with_hulls(monkeypatch: pytest.MonkeyPatch, volumes: list[float]) -> None:
    hulls = []

    def _trimesh(_vertices, _faces):
        hull = _FakeMesh()
        hull.volume = volumes[len(hulls)]
        hulls.append(hull)
        return hull

    _patch_module(monkeypatch, "trimesh", types.SimpleNamespace(Scene=_FakeScene, Trimesh=_trimesh))


def test_collision_build_keeps_an_accurate_single_hull(monkeypatch: pytest.MonkeyPatch):
    """CoACD must not damage an object whose convex hull already fits its scan."""
    events: list[tuple] = []
    _fake_coacd(monkeypatch, events)
    monkeypatch.setattr(
        ycb_assets,
        "_hull_surface_gap",
        lambda _hull, _mesh: (
            ycb_assets._HULL_GAP_GATE_M,
            ycb_assets._HULL_GAP_MAX_GATE_M,
        ),
    )
    mesh = _ConcaveMesh()

    build = ycb_assets._build_collision_geometry(mesh)

    assert build.parts == (mesh.convex_hull,)
    assert not build.decomposed
    assert build.settings is None
    assert events == []


def test_collision_build_uses_the_audited_coacd_configuration(
    monkeypatch: pytest.MonkeyPatch,
):
    """A poor hull fit must use uncapped parts whose vertices match MuJoCo's limit."""
    events: list[tuple] = []
    _fake_coacd(monkeypatch, events)
    _fake_trimesh_with_hulls(monkeypatch, [1.0, 3.0])
    monkeypatch.setattr(
        ycb_assets,
        "_hull_surface_gap",
        lambda _hull, _mesh: (ycb_assets._HULL_GAP_GATE_M + 0.001, 0.0),
    )

    build = ycb_assets._build_collision_geometry(_ConcaveMesh())

    assert build.decomposed
    assert build.settings == ycb_assets._COACD_SETTINGS
    assert events[1] == ("run_coacd", ycb_assets._COACD_SETTINGS)
    assert build.settings["max_convex_hull"] == -1
    assert build.settings["decimate"] is True
    assert build.settings["max_ch_vertex"] == 128


@pytest.mark.parametrize("missing", ["coacd", "threadpoolctl", "rtree"])
def test_collision_build_falls_back_to_single_hull_without_the_extra(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, missing: str
):
    """The decomposer is optional, so a plain install must still build scenes."""
    import logging

    _fake_coacd(monkeypatch, [])
    _patch_module(monkeypatch, missing, None)
    mesh = _ConcaveMesh()

    with caplog.at_level(logging.WARNING, logger="so101_nexus.ycb_assets"):
        build = ycb_assets._build_collision_geometry(mesh)

    assert build.parts == (mesh.convex_hull,)
    assert build.decomposer == ycb_assets._FALLBACK_DECOMPOSER
    assert missing in caplog.records[0].getMessage()


def test_write_collision_parts_records_provenance_and_mass(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    events: list[tuple] = []
    _fake_coacd(monkeypatch, events)
    _fake_trimesh_with_hulls(monkeypatch, [1.0, 3.0])
    monkeypatch.setattr(ycb_assets, "_hull_surface_gap", lambda _h, _m: (0.004, 0.012))
    source = tmp_path / "visual.obj"
    source.write_text("visual", encoding="utf-8")
    out_dir = tmp_path / "collision_v3"

    ycb_assets._write_collision_parts(
        _ConcaveMesh(),
        out_dir,
        model_id="037_scissors",
        source_path=source,
    )

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert [part["mass_fraction"] for part in manifest["parts"]] == [0.25, 0.75]
    assert sum(part["mass_fraction"] for part in manifest["parts"]) == 1.0
    assert manifest["model_id"] == "037_scissors"
    assert manifest["source_sha256"] == ycb_assets._sha256(source)
    assert manifest["settings"] == ycb_assets._COACD_SETTINGS
    assert [part["n_vertices"] for part in manifest["parts"]] == [4, 4]
    assert events[0] == ("enter_limits", 1, "openmp")
    assert events[1] == ("run_coacd", ycb_assets._COACD_SETTINGS)
    assert events[2] == ("exit_limits",)


def test_write_collision_parts_drops_stale_parts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """A shorter decomposition must not leave orphan OBJs behind in the cache."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "collision_007.obj").write_text("stale", encoding="utf-8")
    source = tmp_path / "visual.obj"
    source.write_text("visual", encoding="utf-8")
    _fake_coacd(monkeypatch, [])
    monkeypatch.setattr(ycb_assets, "_hull_surface_gap", lambda _h, _m: (0.0, 0.0))

    ycb_assets._write_collision_parts(
        _FakeMesh(),
        tmp_path,
        model_id="058_golf_ball",
        source_path=source,
    )

    assert sorted(p.name for p in tmp_path.glob("collision_*.obj")) == ["collision_000.obj"]


def _exterior_collision_points(trimesh, parts, count: int) -> np.ndarray:
    """Sample only the exterior faces of abutting convex parts."""
    areas = np.asarray([part.area for part in parts], dtype=np.float64)
    counts = np.maximum((count * areas / areas.sum()).astype(np.int64), 1)
    planes = []
    for part in parts:
        hull = part.convex_hull
        normals = np.asarray(hull.face_normals, dtype=np.float64)
        offsets = np.einsum("ij,ij->i", normals, np.asarray(hull.triangles)[:, 0])
        planes.append((normals, offsets))

    kept = []
    for part_index, (part, sample_count) in enumerate(zip(parts, counts, strict=True)):
        points, face_ids = trimesh.sample.sample_surface(part, int(sample_count), seed=0)
        points = np.asarray(points, dtype=np.float64)
        probes = points + 5e-4 * np.asarray(part.face_normals)[face_ids]
        interior = np.zeros(len(points), dtype=bool)
        for other_index, (normals, offsets) in enumerate(planes):
            if other_index != part_index:
                interior |= ((probes @ normals.T) - offsets <= 0.0).all(axis=1)
        kept.append(points[~interior])
    return np.concatenate(kept)


@pytest.mark.slow
@pytest.mark.parametrize(
    "model_id,decomposed",
    [
        ("009_gelatin_box", False),
        ("011_banana", True),
        ("030_fork", True),
        ("031_spoon", True),
        ("032_knife", True),
        ("033_spatula", True),
        ("037_scissors", True),
        ("040_large_marker", False),
        ("043_phillips_screwdriver", True),
        ("058_golf_ball", False),
    ],
)
def test_collision_geometry_tracks_the_visual_surface(model_id: str, decomposed: bool):
    """The collision surface must not add more than 3.5 mm of bulk error."""
    pytest.importorskip("coacd")
    pytest.importorskip("rtree")
    trimesh = pytest.importorskip("trimesh")

    ycb_assets.ensure_ycb_assets(model_id)
    visual = trimesh.load(str(ycb_assets.get_ycb_visual_mesh(model_id)), force="mesh")
    parts = [
        trimesh.load(str(path), force="mesh")
        for path in ycb_assets.get_ycb_collision_meshes(model_id)
    ]
    visual.update_faces(visual.nondegenerate_faces())
    points = _exterior_collision_points(trimesh, parts, ycb_assets._HULL_GAP_SAMPLES)
    distances = trimesh.proximity.closest_point(visual, points)[1]
    p95 = float(np.percentile(distances, 95))
    maximum = float(np.max(distances))

    manifest = ycb_assets._read_manifest(ycb_assets._collision_dir(model_id))
    assert manifest is not None
    assert manifest["decomposed"] is decomposed
    assert (len(parts) > 1) is decomposed
    assert sum(part.mass_fraction for part in ycb_assets.get_ycb_collision_parts(model_id)) == (
        pytest.approx(1.0)
    )
    if decomposed:
        assert max(len(part.vertices) for part in parts) <= 128
    assert p95 <= 0.0035, f"{model_id} adds {p95 * 1000:.2f} mm of p95 collision error"
    assert maximum <= 0.016, f"{model_id} adds {maximum * 1000:.2f} mm of maximum collision error"
