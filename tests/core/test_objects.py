import pytest

from so101_nexus.objects import (
    CubeObject,
    CylinderObject,
    GSOObject,
    MeshObject,
    PrimitiveObject,
    PyramidObject,
    ScannedMeshObject,
    SceneObject,
    SphereObject,
    YCBObject,
)

PRIMITIVES = (
    (CubeObject, "cube", "side_length_mm"),
    (CylinderObject, "cylinder", "diameter_mm"),
    (SphereObject, "sphere", "diameter_mm"),
    (PyramidObject, "pyramid", "side_length_mm"),
)
SCANNED_OBJECTS = (
    (YCBObject, "009_gelatin_box", "gelatin box", "030_fork", "fork", "999_unknown"),
    (GSOObject, "CoQ10", "supplement bottle", "Pony_C_Clamp_1440", "C-clamp", "Unknown_Model"),
)


def test_object_bases_are_abstract():
    for object_type in (SceneObject, PrimitiveObject, ScannedMeshObject):
        with pytest.raises(TypeError):
            object_type()


def test_primitive_construction_and_descriptions():
    for object_type, shape_name, dimension in PRIMITIVES:
        default = object_type()
        assert isinstance(default, SceneObject)
        assert default.half_size == 0.0125
        assert default.mass == 0.01
        assert default.color == "red"
        assert repr(default) == f"red {shape_name}"
        assert repr(object_type(color="green")) == f"green {shape_name}"
        for length_mm in (0.5, 25.4):
            assert object_type(**{dimension: length_mm}).half_size == pytest.approx(
                length_mm / 2000
            )

        custom = object_type(half_size=0.025, mass=0.2, color="blue")
        assert custom.half_size == 0.025
        assert custom.mass == 0.2
        assert custom.color == "blue"
        assert repr(custom) == f"blue {shape_name}"


def test_primitive_validation():
    for object_type, _, dimension in PRIMITIVES:
        for value in (0.0, -0.01):
            with pytest.raises(ValueError, match="half_size must be positive"):
                object_type(half_size=value)
            with pytest.raises(ValueError, match="mass must be positive"):
                object_type(mass=value)
            with pytest.raises(ValueError, match=f"{dimension} must be positive"):
                object_type(**{dimension: value})
        with pytest.raises(ValueError, match="color must be one of"):
            object_type(color="fuschia")
        with pytest.raises(ValueError, match=f"either half_size or {dimension}"):
            object_type(half_size=0.0125, **{dimension: 25.0})


def test_scanned_object_construction_and_descriptions():
    for object_type, model_id, description, other_id, other_description, _ in SCANNED_OBJECTS:
        obj = object_type(model_id=model_id)
        assert isinstance(obj, SceneObject)
        assert isinstance(obj, ScannedMeshObject)
        assert obj.model_id == model_id
        assert obj.mass_override is None
        assert repr(obj) == description
        assert repr(object_type(model_id=other_id)) == other_description
        overridden = object_type(model_id=model_id, mass_override=0.25)
        assert overridden.model_id == model_id
        assert overridden.mass_override == 0.25


def test_scanned_object_validation():
    for object_type, model_id, _, _, _, invalid_id in SCANNED_OBJECTS:
        with pytest.raises(TypeError):
            object_type()
        with pytest.raises(ValueError, match="model_id must be one of"):
            object_type(model_id=invalid_id)
        for value in (0.0, -1.0):
            with pytest.raises(ValueError, match="mass_override must be positive"):
                object_type(model_id=model_id, mass_override=value)


def test_mesh_construction_and_description():
    default = MeshObject(
        collision_mesh_path="/a.obj", visual_mesh_path="/b.obj", mass=0.1, name="widget"
    )
    assert isinstance(default, SceneObject)
    assert default.collision_mesh_path == "/a.obj"
    assert default.visual_mesh_path == "/b.obj"
    assert default.mass == 0.1
    assert default.name == "widget"
    assert default.scale == 1.0
    assert repr(default) == "widget"

    custom = MeshObject("/collision.stl", "/visual.stl", 0.25, "wrench", scale=2.0)
    assert custom.collision_mesh_path == "/collision.stl"
    assert custom.visual_mesh_path == "/visual.stl"
    assert custom.mass == 0.25
    assert custom.name == "wrench"
    assert custom.scale == 2.0
    assert repr(custom) == "wrench"


def test_mesh_validation():
    for field in ("mass", "scale"):
        for value in (0.0, -1.0):
            kwargs = {"mass": 0.1, "scale": 1.0, field: value}
            with pytest.raises(ValueError, match=f"{field} must be positive"):
                MeshObject(
                    collision_mesh_path="/a.obj", visual_mesh_path="/b.obj", name="x", **kwargs
                )
