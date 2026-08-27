from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

from so101_nexus import gso_assets, mesh_assets
from so101_nexus.constants import GSO_MASSES, GSO_OBJECTS
from so101_nexus.gso_assets import get_gso_mesh_dir

EXPECTED_MODEL_IDS = list(GSO_OBJECTS)


class TestGSOConstants:
    def test_gso_objects_values_are_strings(self):
        for model_id, name in GSO_OBJECTS.items():
            assert isinstance(name, str), f"{model_id} name is not a string"
            assert len(name) > 0, f"{model_id} name is empty"

    def test_every_object_has_a_mass(self):
        assert set(GSO_MASSES) == set(GSO_OBJECTS)
        for model_id, mass in GSO_MASSES.items():
            assert mass > 0, f"{model_id} mass is not positive"


class TestGSOAssets:
    def test_get_gso_mesh_dir_returns_path(self):
        result = get_gso_mesh_dir("Pony_C_Clamp_1440")
        assert isinstance(result, Path)

    def test_get_gso_mesh_dir_invalid_model_raises(self):
        with pytest.raises(ValueError, match="model_id"):
            get_gso_mesh_dir("invalid_model")

    def test_get_gso_mesh_dir_contains_model_id(self):
        for model_id in EXPECTED_MODEL_IDS:
            path = get_gso_mesh_dir(model_id)
            assert model_id in str(path)


def _patch_module(monkeypatch: pytest.MonkeyPatch, name: str, module: object) -> None:
    import sys

    monkeypatch.setitem(sys.modules, name, module)


def test_get_gso_texture_file_returns_cache_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(gso_assets, "_CACHE_DIR", tmp_path)

    assert gso_assets.get_gso_texture_file("CoQ10") == tmp_path / "CoQ10" / "texture.png"


def _write_cached_parts(mesh_dir: Path, files: tuple[str, ...] = ("collision_000.obj",)) -> Path:
    """Write a collision-v3 cache entry and return its directory."""
    mesh_dir.mkdir(parents=True, exist_ok=True)
    source = mesh_dir / "visual.obj"
    source.write_text("v", encoding="utf-8")
    parts_dir = mesh_dir / "collision_v3"
    parts_dir.mkdir(parents=True, exist_ok=True)
    for name in files:
        (parts_dir / name).write_text("c", encoding="utf-8")
    manifest = {
        "model_id": mesh_dir.name,
        "source": source.name,
        "source_sha256": mesh_assets._sha256(source),
        "decomposer": mesh_assets._HULL_DECOMPOSER,
        "settings": None,
        "hull_gap_p95_m": 0.0,
        "hull_gap_max_m": 0.0,
        "hull_gap_gate_m": mesh_assets._HULL_GAP_GATE_M,
        "hull_gap_max_gate_m": mesh_assets._HULL_GAP_MAX_GATE_M,
        "decomposed": False,
        "parts": [{"file": name, "mass_fraction": 1.0 / len(files)} for name in files],
    }
    (parts_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return parts_dir


def test_ensure_gso_assets_cache_hit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(gso_assets, "_CACHE_DIR", tmp_path)
    model_id = "Pony_C_Clamp_1440"
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

    assert gso_assets.ensure_gso_assets(model_id) == mesh_dir


def test_ensure_gso_assets_cache_hit_copies_missing_texture(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setattr(gso_assets, "_CACHE_DIR", tmp_path)
    model_id = "Pony_C_Clamp_1440"
    mesh_dir = tmp_path / model_id
    mesh_dir.mkdir(parents=True)
    _write_cached_parts(mesh_dir)
    (mesh_dir / "visual.obj").write_text("v", encoding="utf-8")
    downloaded_texture = tmp_path / "meshes" / model_id / "texture.png"
    downloaded_texture.parent.mkdir(parents=True)
    downloaded_texture.write_text("texture-bytes", encoding="utf-8")

    def _unexpected_snapshot_download(**_kwargs):
        raise AssertionError("a cached mirror texture should not trigger a download")

    _patch_module(
        monkeypatch,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=_unexpected_snapshot_download),
    )

    result = gso_assets.ensure_gso_assets(model_id)

    assert result == mesh_dir
    assert (mesh_dir / "texture.png").read_text(encoding="utf-8") == "texture-bytes"


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


def _fake_trimesh(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_module(
        monkeypatch,
        "trimesh",
        types.SimpleNamespace(
            Scene=type("_FakeScene", (), {}),
            load=lambda *_a, **_k: _FakeMesh(),
        ),
    )


def test_ensure_gso_assets_warns_without_a_mirrored_texture(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    import logging

    model_id = "Pony_C_Clamp_1440"
    monkeypatch.setattr(gso_assets, "_CACHE_DIR", tmp_path)

    def _snapshot_download(**_kwargs):
        obj_dir = tmp_path / "meshes" / model_id
        obj_dir.mkdir(parents=True, exist_ok=True)
        (obj_dir / "model.obj").write_text("v", encoding="utf-8")

    _patch_module(
        monkeypatch,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=_snapshot_download),
    )
    _fake_coacd(monkeypatch, [])
    _fake_trimesh(monkeypatch)
    monkeypatch.setattr(mesh_assets, "_hull_surface_gap", lambda _h, _m: (0.0, 0.0))

    with caplog.at_level(logging.WARNING, logger="so101_nexus.gso_assets"):
        mesh_dir = gso_assets.ensure_gso_assets(model_id)

    assert mesh_dir == tmp_path / model_id
    assert not (mesh_dir / "texture.png").exists()
    assert any(model_id in r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)


def _fake_coacd(monkeypatch: pytest.MonkeyPatch, events: list[tuple]) -> None:
    """Install fake ``coacd``/``threadpoolctl``/``rtree`` modules recording their use."""
    import contextlib

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


def test_ensure_gso_assets_download_copies_obj_directly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """GSO ships ready-to-use OBJ + PNG, unlike YCB's GLB conversion path."""
    model_id = "Pony_C_Clamp_1440"
    monkeypatch.setattr(gso_assets, "_CACHE_DIR", tmp_path)

    def _snapshot_download(**kwargs):
        assert kwargs["allow_patterns"] == [f"meshes/{model_id}/*"]
        obj_dir = tmp_path / "meshes" / model_id
        obj_dir.mkdir(parents=True, exist_ok=True)
        (obj_dir / "model.obj").write_text("real-world-scale obj", encoding="utf-8")
        (obj_dir / "texture.png").write_text("texture-bytes", encoding="utf-8")

    _patch_module(
        monkeypatch,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=_snapshot_download),
    )
    _fake_coacd(monkeypatch, [])
    _fake_trimesh(monkeypatch)
    monkeypatch.setattr(mesh_assets, "_hull_surface_gap", lambda _h, _m: (0.0, 0.0))

    mesh_dir = gso_assets.ensure_gso_assets(model_id)

    assert mesh_dir == tmp_path / model_id
    assert (mesh_dir / "visual.obj").read_text(encoding="utf-8") == "real-world-scale obj"
    assert (mesh_dir / "texture.png").read_text(encoding="utf-8") == "texture-bytes"


def test_collision_and_visual_mesh_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(gso_assets, "_CACHE_DIR", tmp_path)
    model_id = "CoQ10"
    parts_dir = _write_cached_parts(tmp_path / model_id, ("collision_000.obj", "collision_001.obj"))

    assert gso_assets.get_gso_collision_meshes(model_id) == [
        parts_dir / "collision_000.obj",
        parts_dir / "collision_001.obj",
    ]
    assert gso_assets.get_gso_collision_mesh(model_id) == parts_dir / "collision_000.obj"
    assert gso_assets.get_gso_visual_mesh(model_id) == tmp_path / model_id / "visual.obj"


def test_collision_parts_require_prepared_assets(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(gso_assets, "_CACHE_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="ensure_gso_assets"):
        gso_assets.get_gso_collision_parts("CoQ10")


def test_source_repo_env_override(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SO101_GSO_HF_REPO", "your-org/your-gso-repo")
    import importlib

    reloaded = importlib.reload(gso_assets)
    try:
        assert reloaded._HF_REPO_ID == "your-org/your-gso-repo"
    finally:
        monkeypatch.delenv("SO101_GSO_HF_REPO", raising=False)
        importlib.reload(gso_assets)
