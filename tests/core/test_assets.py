"""Tests for asset-path accessors."""

import time
from concurrent.futures import ThreadPoolExecutor

from so101_nexus import (
    get_so101_mujoco_model_dir,
    get_so101_mujoco_model_path,
    mesh_assets,
)


def test_mujoco_model_path_points_at_menagerie():
    d = get_so101_mujoco_model_dir()
    p = get_so101_mujoco_model_path()
    assert d.name == "SO101_menagerie"
    assert p == d / "so101.xml"
    assert p.is_file()
    assert (d / "assets").is_dir()


def test_scanned_asset_build_is_locked_per_model(monkeypatch, tmp_path):
    mesh_dir = tmp_path / "model"
    collision_dir = mesh_dir / "collision_v3"
    visual_path = mesh_dir / "visual.obj"
    texture_path = mesh_dir / "texture.png"
    built = False
    calls = {"fetch": 0, "write": 0}

    monkeypatch.setattr(mesh_assets, "_collision_parts_are_current", lambda _path: built)
    monkeypatch.setattr(mesh_assets, "_load_exportable_mesh", lambda _path: object())

    def fetch():
        calls["fetch"] += 1
        mesh_dir.mkdir(exist_ok=True)
        visual_path.write_text("mesh", encoding="utf-8")
        time.sleep(0.05)

    def write(*_args, **_kwargs):
        nonlocal built
        calls["write"] += 1
        built = True

    monkeypatch.setattr(mesh_assets, "_write_collision_parts", write)

    def ensure():
        return mesh_assets.ensure_scanned_mesh_assets(
            model_id="model",
            mesh_dir=mesh_dir,
            collision_dir=collision_dir,
            visual_path=visual_path,
            texture_path=texture_path,
            fetch=fetch,
            ensure_texture=lambda: None,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _index: ensure(), range(2)))

    assert results == [mesh_dir, mesh_dir]
    assert calls == {"fetch": 1, "write": 1}
