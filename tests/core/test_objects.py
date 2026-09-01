import pytest

from so101_nexus.objects import (
    CubeObject,
    CylinderObject,
    GSOObject,
    MeshObject,
    PyramidObject,
    ScannedMeshObject,
    SceneObject,
    SphereObject,
    YCBObject,
)


class TestSceneObjectBase:
    def test_cube_is_scene_object(self):
        assert isinstance(CubeObject(), SceneObject)

    def test_ycb_is_scene_object(self):
        assert isinstance(YCBObject(model_id="009_gelatin_box"), SceneObject)
        assert isinstance(YCBObject(model_id="009_gelatin_box"), ScannedMeshObject)

    def test_gso_is_scene_object(self):
        assert isinstance(GSOObject(model_id="CoQ10"), SceneObject)
        assert isinstance(GSOObject(model_id="CoQ10"), ScannedMeshObject)

    def test_mesh_is_scene_object(self):
        obj = MeshObject(
            collision_mesh_path="/a.obj", visual_mesh_path="/b.obj", mass=0.1, name="x"
        )
        assert isinstance(obj, SceneObject)

    def test_cannot_instantiate_base(self):
        with pytest.raises(TypeError):
            SceneObject()

    def test_all_have_repr(self):
        objects: list[SceneObject] = [
            CubeObject(),
            CylinderObject(),
            SphereObject(),
            PyramidObject(),
            YCBObject(model_id="009_gelatin_box"),
            GSOObject(model_id="CoQ10"),
            MeshObject(
                collision_mesh_path="/a.obj", visual_mesh_path="/b.obj", mass=0.1, name="widget"
            ),
        ]
        for obj in objects:
            assert isinstance(repr(obj), str)
            assert repr(obj)


class TestCubeObject:
    def test_defaults(self):
        obj = CubeObject()
        assert obj.half_size > 0
        assert obj.mass > 0

    def test_repr_color_cube(self):
        assert repr(CubeObject(color="red")) == "red cube"
        assert repr(CubeObject(color="green")) == "green cube"

    def test_invalid_size(self):
        with pytest.raises(ValueError, match="half_size must be positive"):
            CubeObject(half_size=-0.01)

    def test_invalid_color(self):
        with pytest.raises(ValueError, match="color must be one of"):
            CubeObject(color="fuschia")

    @pytest.mark.parametrize(
        ("object_type", "dimension"),
        [
            (CubeObject, "side_length_mm"),
            (CylinderObject, "diameter_mm"),
            (SphereObject, "diameter_mm"),
            (PyramidObject, "side_length_mm"),
        ],
    )
    def test_converts_millimeter_dimensions(self, object_type, dimension):
        obj = object_type(**{dimension: 25.4})

        assert obj.half_size == pytest.approx(0.0127)

    def test_rejects_ambiguous_size_inputs(self):
        with pytest.raises(ValueError, match="either half_size or side_length_mm"):
            CubeObject(half_size=0.0125, side_length_mm=25.0)


@pytest.mark.parametrize(
    ("object_type", "shape_name"),
    [
        (CylinderObject, "cylinder"),
        (SphereObject, "sphere"),
        (PyramidObject, "pyramid"),
    ],
)
class TestShapeObjects:
    def test_defaults(self, object_type, shape_name):
        obj = object_type()
        assert obj.half_size > 0
        assert obj.mass > 0
        assert repr(obj) == f"red {shape_name}"

    def test_accepts_cube_colors(self, object_type, shape_name):
        assert repr(object_type(color="green")) == f"green {shape_name}"

    def test_rejects_invalid_size(self, object_type, shape_name):
        with pytest.raises(ValueError, match="half_size must be positive"):
            object_type(half_size=-0.01)

    def test_rejects_invalid_color(self, object_type, shape_name):
        with pytest.raises(ValueError, match="color must be one of"):
            object_type(color="fuschia")


class TestYCBObject:
    def test_requires_model_id(self):
        with pytest.raises(TypeError):
            YCBObject()

    def test_valid(self):
        obj = YCBObject(model_id="009_gelatin_box")
        assert obj.model_id == "009_gelatin_box"

    def test_repr_human_readable(self):
        assert repr(YCBObject(model_id="009_gelatin_box")) == "gelatin box"
        assert repr(YCBObject(model_id="030_fork")) == "fork"

    def test_invalid_model_id(self):
        with pytest.raises(ValueError, match="model_id must be one of"):
            YCBObject(model_id="999_unknown")

    def test_invalid_mass_override(self):
        with pytest.raises(ValueError, match="mass_override must be positive"):
            YCBObject(model_id="009_gelatin_box", mass_override=-1.0)

    def test_valid_mass_override(self):
        obj = YCBObject(model_id="009_gelatin_box", mass_override=0.25)
        assert obj.mass_override == 0.25


class TestGSOObject:
    def test_requires_model_id(self):
        with pytest.raises(TypeError):
            GSOObject()

    def test_valid(self):
        obj = GSOObject(model_id="CoQ10")
        assert obj.model_id == "CoQ10"

    def test_repr_human_readable(self):
        assert repr(GSOObject(model_id="CoQ10")) == "supplement bottle"
        assert repr(GSOObject(model_id="Pony_C_Clamp_1440")) == "C-clamp"

    def test_invalid_model_id(self):
        with pytest.raises(ValueError, match="model_id must be one of"):
            GSOObject(model_id="Unknown_Model")

    def test_invalid_mass_override(self):
        with pytest.raises(ValueError, match="mass_override must be positive"):
            GSOObject(model_id="CoQ10", mass_override=-1.0)

    def test_valid_mass_override(self):
        obj = GSOObject(model_id="CoQ10", mass_override=0.25)
        assert obj.mass_override == 0.25


class TestMeshObject:
    def test_valid(self):
        obj = MeshObject(
            collision_mesh_path="/a.obj",
            visual_mesh_path="/b.obj",
            mass=0.1,
            name="widget",
        )
        assert obj.mass == 0.1

    def test_repr_uses_name(self):
        obj = MeshObject(
            collision_mesh_path="/a.obj", visual_mesh_path="/b.obj", mass=0.1, name="wrench"
        )
        assert repr(obj) == "wrench"

    def test_invalid_mass(self):
        with pytest.raises(ValueError, match="mass must be positive"):
            MeshObject(collision_mesh_path="/a.obj", visual_mesh_path="/b.obj", mass=-1.0, name="x")
