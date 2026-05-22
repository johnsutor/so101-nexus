from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from so101_nexus_core import ycb_assets
from so101_nexus_core.constants import YCB_OBJECTS
from so101_nexus_core.ycb_assets import get_ycb_mesh_dir

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
    def __init__(self):
        self.exports: list[tuple[str, str | None]] = []
        self.convex_hull = self

    def export(self, file_obj: str, file_type: str | None = None):
        self.exports.append((file_obj, file_type))


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


def test_ensure_ycb_assets_cache_hit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(ycb_assets, "_CACHE_DIR", tmp_path)
    model_id = "009_gelatin_box"
    mesh_dir = tmp_path / model_id
    mesh_dir.mkdir(parents=True)
    (mesh_dir / "collision.obj").write_text("c", encoding="utf-8")
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
    (mesh_dir / "collision.obj").write_text("c", encoding="utf-8")
    (mesh_dir / "visual.obj").write_text("v", encoding="utf-8")
    glb = tmp_path / "meshes" / model_id / "google_16k" / "textured.glb"
    glb.parent.mkdir(parents=True)
    glb.write_text("fake-glb", encoding="utf-8")
    calls: list[tuple[Path, Path]] = []

    def _extract(glb_path: Path, texture_path: Path) -> bool:
        calls.append((glb_path, texture_path))
        texture_path.write_text("texture", encoding="utf-8")
        return True

    monkeypatch.setattr(ycb_assets, "_extract_glb_texture", _extract)

    result = ycb_assets.ensure_ycb_assets(model_id)

    assert result == mesh_dir
    assert calls == [(glb, mesh_dir / "texture.png")]


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

    mesh_dir = ycb_assets.ensure_ycb_assets(model_id)

    assert mesh_dir == tmp_path / model_id
    assert not (mesh_dir / "texture.png").exists()


def test_ensure_ycb_assets_download_and_convex_hull(
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

    mesh_dir = ycb_assets.ensure_ycb_assets(model_id)
    assert mesh_dir == tmp_path / model_id
    assert "snapshot_kwargs" in called
    assert called["convert_args"] == (
        tmp_path / "meshes" / model_id / "google_16k" / "textured.glb",
        tmp_path / model_id / "visual.obj",
    )
    assert mesh.exports[-1] == (str(tmp_path / model_id / "collision.obj"), "obj")


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

    ycb_assets.ensure_ycb_assets(model_id)
    assert mesh.exports[-1] == (str(tmp_path / model_id / "collision.obj"), "obj")


def test_collision_and_visual_mesh_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(ycb_assets, "_CACHE_DIR", tmp_path)
    model_id = "058_golf_ball"
    assert ycb_assets.get_ycb_collision_mesh(model_id) == tmp_path / model_id / "collision.obj"
    assert ycb_assets.get_ycb_visual_mesh(model_id) == tmp_path / model_id / "visual.obj"


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
