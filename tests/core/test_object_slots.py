"""Tests for the shared object-slot abstraction (backend-neutral)."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco

from so101_nexus import get_so101_mujoco_model_path
from so101_nexus.object_slots import (
    build_object_scene_xml,
    cube_bounding_radius,
    cube_xml_body,
    extract_object_slots,
    mesh_xml_body,
    object_bounding_radius,
    primary_geom_name,
)
from so101_nexus.objects import CubeObject, MeshObject, YCBObject
from so101_nexus.scene import MUJOCO_SCENE_OPTION_XML

_ROBOT_XML = str(get_so101_mujoco_model_path())


def _cube_scene(objects):
    slot_names = [f"pick_slot_{i}" for i in range(len(objects))]
    return (
        build_object_scene_xml(
            objects,
            slot_names,
            [0.5, 0.5, 0.5, 1.0],
            option_xml=MUJOCO_SCENE_OPTION_XML,
            robot_xml_path=_ROBOT_XML,
        ),
        slot_names,
    )


class TestXmlBuilders:
    def test_cube_xml_body_fields(self):
        xml = cube_xml_body("pick_slot_0", CubeObject(half_size=0.02, mass=0.03, color="blue"))
        assert 'name="pick_slot_0"' in xml
        assert 'type="box" size="0.02 0.02 0.02"' in xml
        assert 'mass="0.03"' in xml
        assert "0.0 0.0 1.0 1.0" in xml  # blue rgba

    def test_mesh_xml_body_groups(self):
        xml = mesh_xml_body("pick_slot_0", 0, 0.01)
        assert 'name="pick_slot_0_collision"' in xml
        assert 'mesh="pick_coll_0"' in xml
        assert 'group="3"' in xml
        assert 'name="pick_slot_0_visual"' in xml
        assert 'mesh="pick_vis_0"' in xml
        assert 'group="2"' in xml

    def test_primary_geom_name(self):
        assert primary_geom_name("pick_slot_0", CubeObject()) == "pick_slot_0_geom"
        assert primary_geom_name("pick_slot_1", YCBObject("011_banana")) == "pick_slot_1_collision"

    def test_cube_bounding_radius(self):
        assert cube_bounding_radius(CubeObject(half_size=0.0125)) == pytest.approx(
            0.0125 * np.sqrt(2)
        )

    def test_object_bounding_radius_cube(self):
        radius = object_bounding_radius(CubeObject(half_size=0.02))
        assert radius == pytest.approx(0.02 * np.sqrt(2))

    def test_build_scene_xml_uses_warp_option(self):
        from so101_nexus.scene import WARP_SCENE_OPTION_XML

        xml = build_object_scene_xml(
            [CubeObject()],
            ["pick_slot_0"],
            [0.5, 0.5, 0.5, 1.0],
            option_xml=WARP_SCENE_OPTION_XML,
            robot_xml_path=_ROBOT_XML,
        )
        assert "noslip" not in xml
        assert 'integrator="implicit"' in xml

    def test_scene_xml_leaves_cubes_and_mesh_objects_untextured(self, tmp_path):
        mesh = MeshObject(
            collision_mesh_path=str(tmp_path / "collision.obj"),
            visual_mesh_path=str(tmp_path / "visual.obj"),
            mass=0.02,
            name="custom",
        )
        xml, _ = _cube_scene([CubeObject(color="red"), mesh])
        assert "<texture " not in xml
        assert 'material="pick_mat_' not in xml

    def test_ycb_scene_xml_binds_cached_texture(self, monkeypatch, tmp_path):
        from so101_nexus import object_slots

        collision_path = tmp_path / "collision.obj"
        visual_path = tmp_path / "visual.obj"
        texture_path = tmp_path / "texture.png"
        texture_path.write_text("texture", encoding="utf-8")
        monkeypatch.setattr(object_slots, "get_ycb_collision_mesh", lambda _m: collision_path)
        monkeypatch.setattr(object_slots, "get_ycb_visual_mesh", lambda _m: visual_path)
        monkeypatch.setattr(object_slots, "get_ycb_texture_file", lambda _m: texture_path)

        xml = build_object_scene_xml(
            [YCBObject(model_id="011_banana")],
            ["pick_slot_0"],
            [0.1, 0.2, 0.3, 1.0],
            option_xml=MUJOCO_SCENE_OPTION_XML,
            robot_xml_path=_ROBOT_XML,
        )
        assert f'<texture name="pick_tex_0" type="2d" file="{texture_path}"/>' in xml
        assert '<material name="pick_mat_0" texture="pick_tex_0" texuniform="false"/>' in xml
        assert 'material="pick_mat_0"' in xml

    def test_ycb_scene_xml_uses_posix_paths_for_cached_assets(self, monkeypatch):
        from so101_nexus import object_slots

        class _FakePath:
            def __init__(self, posix_path: str, native_path: str):
                self._posix_path = posix_path
                self._native_path = native_path

            def __str__(self) -> str:
                return self._native_path

            def as_posix(self) -> str:
                return self._posix_path

            def exists(self) -> bool:
                return True

        collision_path = _FakePath("C:/cache/ycb/collision.obj", r"C:\cache\ycb\collision.obj")
        visual_path = _FakePath("C:/cache/ycb/visual.obj", r"C:\cache\ycb\visual.obj")
        texture_path = _FakePath("C:/cache/ycb/texture.png", r"C:\cache\ycb\texture.png")
        monkeypatch.setattr(object_slots, "get_ycb_collision_mesh", lambda _m: collision_path)
        monkeypatch.setattr(object_slots, "get_ycb_visual_mesh", lambda _m: visual_path)
        monkeypatch.setattr(object_slots, "get_ycb_texture_file", lambda _m: texture_path)

        xml = build_object_scene_xml(
            [YCBObject(model_id="011_banana")],
            ["pick_slot_0"],
            [0.1, 0.2, 0.3, 1.0],
            option_xml=MUJOCO_SCENE_OPTION_XML,
            robot_xml_path=_ROBOT_XML,
        )
        assert 'file="C:/cache/ycb/collision.obj"' in xml
        assert 'file="C:/cache/ycb/visual.obj"' in xml
        assert 'file="C:/cache/ycb/texture.png"' in xml
        assert r"C:\cache\ycb" not in xml


class TestSlotExtraction:
    def test_extract_cube_slots(self):
        objects = [
            CubeObject(half_size=0.0125, color="red"),
            CubeObject(half_size=0.02, color="blue"),
        ]
        xml, slot_names = _cube_scene(objects)
        so_dir = get_so101_mujoco_model_path().parent
        with tempfile.NamedTemporaryFile("w", suffix=".xml", dir=so_dir, delete=True) as f:
            f.write(xml)
            f.flush()
            mjm = mujoco.MjModel.from_xml_path(f.name)
        slots = extract_object_slots(mjm, slot_names, objects)
        assert len(slots) == 2
        # qpos/dof addresses distinct and 7/6 wide respectively
        assert slots[1].qpos_addr - slots[0].qpos_addr == 7
        assert slots[1].dof_addr - slots[0].dof_addr == 6
        assert slots[0].bounding_radius == pytest.approx(0.0125 * np.sqrt(2))
        assert slots[1].bounding_radius == pytest.approx(0.02 * np.sqrt(2))
        assert slots[0].spawn_z == pytest.approx(0.0125)
        np.testing.assert_allclose(slots[0].rest_quat, [1.0, 0.0, 0.0, 0.0])
        for slot in slots:
            assert slot.geom_id >= 0

    def test_mesh_bounding_radius_uses_rotated_footprint(self, tmp_path):
        # A box thin in X: the stable rest pose rotates X up, so the horizontal
        # footprint is the original Y and Z extents, not the original X and Y.
        hx, hy, hz = 0.005, 0.03, 0.02
        corners = [
            (sx * hx, sy * hy, sz * hz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)
        ]
        faces = [
            (1, 2, 4),
            (1, 4, 3),
            (5, 6, 8),
            (5, 8, 7),
            (1, 2, 6),
            (1, 6, 5),
            (3, 4, 8),
            (3, 8, 7),
            (1, 3, 7),
            (1, 7, 5),
            (2, 4, 8),
            (2, 8, 6),
        ]
        obj_lines = [f"v {x} {y} {z}" for x, y, z in corners]
        obj_lines += [f"f {a} {b} {c}" for a, b, c in faces]
        mesh_path = tmp_path / "thin_box.obj"
        mesh_path.write_text("\n".join(obj_lines) + "\n", encoding="utf-8")

        mesh = MeshObject(
            collision_mesh_path=str(mesh_path),
            visual_mesh_path=str(mesh_path),
            mass=0.02,
            name="thin box",
        )
        xml, slot_names = _cube_scene([mesh])
        so_dir = get_so101_mujoco_model_path().parent
        with tempfile.NamedTemporaryFile("w", suffix=".xml", dir=so_dir, delete=True) as f:
            f.write(xml)
            f.flush()
            mjm = mujoco.MjModel.from_xml_path(f.name)
        slot = extract_object_slots(mjm, slot_names, [mesh])[0]

        rotated_radius = float(np.linalg.norm([2 * hy, 2 * hz]) / 2)
        naive_radius = float(np.linalg.norm([2 * hx, 2 * hy]) / 2)
        assert slot.bounding_radius == pytest.approx(rotated_radius, abs=1e-6)
        assert slot.bounding_radius != pytest.approx(naive_radius, abs=1e-6)
