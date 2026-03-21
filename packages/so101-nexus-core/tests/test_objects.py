import pytest

from so101_nexus_core.objects import CubeObject, MeshObject, SceneObject, YCBObject


class TestSceneObjectBase:
    def test_cube_is_scene_object(self):
        assert isinstance(CubeObject(), SceneObject)

    def test_ycb_is_scene_object(self):
        assert isinstance(YCBObject(model_id="009_gelatin_box"), SceneObject)

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
            YCBObject(model_id="011_banana"),
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


class TestYCBObject:
    def test_requires_model_id(self):
        with pytest.raises(TypeError):
            YCBObject()

    def test_valid(self):
        obj = YCBObject(model_id="009_gelatin_box")
        assert obj.model_id == "009_gelatin_box"

    def test_repr_human_readable(self):
        assert repr(YCBObject(model_id="009_gelatin_box")) == "gelatin box"
        assert repr(YCBObject(model_id="011_banana")) == "banana"

    def test_invalid_model_id(self):
        with pytest.raises(ValueError, match="model_id must be one of"):
            YCBObject(model_id="999_unknown")


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
