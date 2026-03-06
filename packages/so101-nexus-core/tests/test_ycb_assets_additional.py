from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

import so101_nexus_core.ycb_assets as ycb_assets


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


def test_ensure_ycb_assets_cache_hit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(ycb_assets, "_CACHE_DIR", tmp_path)
    model_id = "009_gelatin_box"
    mesh_dir = tmp_path / model_id
    mesh_dir.mkdir(parents=True)
    (mesh_dir / "collision.obj").write_text("c", encoding="utf-8")
    (mesh_dir / "visual.obj").write_text("v", encoding="utf-8")

    def _unexpected_snapshot_download(**_kwargs):
        raise AssertionError("snapshot_download should not be called on cache hit")

    _patch_module(
        monkeypatch,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=_unexpected_snapshot_download),
    )

    result = ycb_assets.ensure_ycb_assets(model_id)
    assert result == mesh_dir


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
