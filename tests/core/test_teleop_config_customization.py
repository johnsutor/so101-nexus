from __future__ import annotations

import inspect

import pytest

from so101_nexus.config import MoveConfig, PickAndPlaceConfig, PickConfig, StackCubeConfig
from so101_nexus.objects import CubeObject, MeshObject, YCBObject
from so101_nexus.teleop.config_customization import (
    TeleopConfigOverrides,
    apply_config_factory,
    apply_config_overrides,
    load_config_factory,
    load_profile_overrides,
    object_from_mapping,
    object_from_spec,
    object_to_mapping,
    overrides_from_mapping,
    overrides_to_mapping,
)


def test_object_from_spec_parses_cube() -> None:
    obj = object_from_spec("cube:blue")

    assert isinstance(obj, CubeObject)
    assert obj.color == "blue"


def test_object_from_spec_parses_ycb() -> None:
    obj = object_from_spec("ycb:011_banana")

    assert isinstance(obj, YCBObject)
    assert obj.model_id == "011_banana"


def test_object_from_mapping_parses_mesh() -> None:
    obj = object_from_mapping(
        {
            "type": "mesh",
            "name": "widget",
            "collision_mesh_path": "/tmp/collision.stl",
            "visual_mesh_path": "/tmp/visual.obj",
            "mass": 0.02,
            "scale": 0.5,
        }
    )

    assert isinstance(obj, MeshObject)
    assert obj.name == "widget"
    assert obj.mass == 0.02
    assert obj.scale == 0.5


def test_object_from_mapping_requires_cube_color() -> None:
    with pytest.raises(ValueError, match="missing required key 'color'"):
        object_from_mapping({"type": "cube"})


def test_object_from_spec_rejects_invalid_color() -> None:
    with pytest.raises(ValueError, match="unknown cube color"):
        object_from_spec("cube:not-a-color")


def test_object_from_spec_rejects_mesh_string_specs() -> None:
    with pytest.raises(ValueError, match="mesh objects must use mapping syntax"):
        object_from_spec("mesh:widget:/tmp/collision.stl:/tmp/visual.obj:0.02:0.5")


def test_apply_config_overrides_updates_pick_objects_and_distractors() -> None:
    cfg = apply_config_overrides(
        PickConfig(),
        TeleopConfigOverrides(object_specs=("cube:green", "ycb:011_banana"), n_distractors=1),
    )

    assert isinstance(cfg, PickConfig)
    assert cfg.n_distractors == 1
    assert isinstance(cfg.objects[0], CubeObject)
    assert cfg.objects[0].color == "green"
    assert isinstance(cfg.objects[1], YCBObject)
    assert cfg.objects[1].model_id == "011_banana"


def test_apply_config_overrides_accepts_parsed_mesh_objects() -> None:
    mesh = MeshObject(
        collision_mesh_path="/tmp/collision.stl",
        visual_mesh_path="/tmp/visual.obj",
        mass=0.02,
        name="widget",
    )

    cfg = apply_config_overrides(PickConfig(), TeleopConfigOverrides(objects=(mesh,)))

    assert cfg.objects == [mesh]


def test_apply_config_overrides_updates_pick_and_place_colors() -> None:
    cfg = apply_config_overrides(
        PickAndPlaceConfig(),
        TeleopConfigOverrides(cube_colors=("red", "green"), target_colors=("blue",)),
    )

    assert cfg.cube_colors == ["red", "green"]
    assert cfg.target_colors == ["blue"]


def test_apply_config_overrides_updates_reset_settle_frames() -> None:
    cfg = apply_config_overrides(PickConfig(), TeleopConfigOverrides(reset_settle_frames=2))

    assert cfg.reset_settle_frames == 2


def test_apply_config_overrides_ignores_pick_specific_options_for_non_pick_config() -> None:
    base = MoveConfig()

    cfg = apply_config_overrides(
        base,
        TeleopConfigOverrides(object_specs=("ycb:011_banana",), n_distractors=1),
    )

    assert isinstance(cfg, MoveConfig)
    assert vars(cfg) == vars(base)


def test_apply_config_overrides_applies_common_fields_to_stack_cube() -> None:
    """StackCubeConfig has no dedicated override keys (unlike cube_colors/target_colors
    for PickAndPlaceConfig), but still picks up common overrides generically."""
    base = StackCubeConfig()

    cfg = apply_config_overrides(
        base,
        TeleopConfigOverrides(ground_colors=("white",), spawn_min_radius=0.2),
    )

    assert isinstance(cfg, StackCubeConfig)
    assert cfg.ground_colors == ["white"]
    assert cfg.spawn_min_radius == 0.2
    assert cfg.cube_a_colors == base.cube_a_colors
    assert cfg.cube_b_colors == base.cube_b_colors


def test_apply_config_overrides_rejects_negative_distractors() -> None:
    with pytest.raises(ValueError, match="n_distractors"):
        apply_config_overrides(PickConfig(), TeleopConfigOverrides(n_distractors=-1))


def test_load_profile_json_merges_common_task_and_env_sections(tmp_path) -> None:
    path = tmp_path / "profile.json"
    path.write_text(
        '{"common":{"ground_colors":["white"]},'
        '"pick":{"objects":[{"type":"cube","color":"green"},'
        '{"type":"ycb","model_id":"011_banana"}],"n_distractors":1},'
        '"envs":{"MuJoCoPickLift-v1":{"spawn_max_radius":0.22}}}',
        encoding="utf-8",
    )

    overrides = load_profile_overrides(path, "MuJoCoPickLift-v1", PickConfig())

    assert overrides.ground_colors == ("white",)
    assert overrides.objects is not None
    assert isinstance(overrides.objects[0], CubeObject)
    assert overrides.objects[0].color == "green"
    assert isinstance(overrides.objects[1], YCBObject)
    assert overrides.objects[1].model_id == "011_banana"
    assert overrides.n_distractors == 1
    assert overrides.spawn_max_radius == 0.22


def test_load_profile_common_accepts_reset_settle_frames(tmp_path) -> None:
    path = tmp_path / "profile.json"
    path.write_text('{"common":{"reset_settle_frames":3}}', encoding="utf-8")

    overrides = load_profile_overrides(path, "MuJoCoPickLift-v1", PickConfig())

    assert overrides.reset_settle_frames == 3


def test_load_profile_precedence_is_env_over_task_over_common(tmp_path) -> None:
    path = tmp_path / "profile.json"
    path.write_text(
        '{"spawn_max_radius":0.05,'
        '"common":{"spawn_max_radius":0.10},'
        '"pick":{"spawn_max_radius":0.20},'
        '"envs":{"MuJoCoPickLift-v1":{"spawn_max_radius":0.30}}}',
        encoding="utf-8",
    )

    overrides = load_profile_overrides(path, "MuJoCoPickLift-v1", PickConfig())

    assert overrides.spawn_max_radius == 0.30


def test_load_profile_toml_supports_pick_and_place(tmp_path) -> None:
    path = tmp_path / "profile.toml"
    path.write_text(
        "[pick_and_place]\ncube_colors=['red','green']\ntarget_colors=['blue']\n",
        encoding="utf-8",
    )

    overrides = load_profile_overrides(path, "MuJoCoPickAndPlace-v1", PickAndPlaceConfig())

    assert overrides.cube_colors == ("red", "green")
    assert overrides.target_colors == ("blue",)


def test_load_profile_accepts_uppercase_suffix(tmp_path) -> None:
    path = tmp_path / "profile.JSON"
    path.write_text('{"common":{"ground_colors":["white"]}}', encoding="utf-8")

    overrides = load_profile_overrides(path, "MuJoCoMove-v1", MoveConfig())

    assert overrides.ground_colors == ("white",)


def test_load_profile_pick_section_does_not_apply_to_pick_and_place(tmp_path) -> None:
    path = tmp_path / "profile.toml"
    path.write_text(
        "[pick]\nn_distractors=2\n"
        "[pick_and_place]\ncube_colors=['green']\ntarget_colors=['blue']\n",
        encoding="utf-8",
    )

    overrides = load_profile_overrides(path, "MuJoCoPickAndPlace-v1", PickAndPlaceConfig())

    assert overrides.n_distractors is None
    assert overrides.cube_colors == ("green",)
    assert overrides.target_colors == ("blue",)


def test_load_profile_rejects_null_env_section(tmp_path) -> None:
    path = tmp_path / "profile.json"
    path.write_text('{"envs":{"MuJoCoPickLift-v1":null}}', encoding="utf-8")

    with pytest.raises(ValueError, match=r"envs\.'MuJoCoPickLift-v1'"):
        load_profile_overrides(path, "MuJoCoPickLift-v1", PickConfig())


def test_load_profile_rejects_invalid_color(tmp_path) -> None:
    path = tmp_path / "profile.json"
    path.write_text('{"pick":{"objects":[{"type":"cube","color":"bad"}]}}', encoding="utf-8")

    with pytest.raises(ValueError, match="unknown cube color"):
        load_profile_overrides(path, "MuJoCoPickLift-v1", PickConfig())


def test_load_profile_rejects_unsupported_suffix(tmp_path) -> None:
    path = tmp_path / "profile.yaml"
    path.write_text("common: {}", encoding="utf-8")

    with pytest.raises(ValueError, match=r"\.json or \.toml"):
        load_profile_overrides(path, "MuJoCoPickLift-v1", PickConfig())


def test_load_profile_rejects_unknown_top_level_override_key(tmp_path) -> None:
    path = tmp_path / "profile.json"
    path.write_text('{"spawn_min_radious":0.1}', encoding="utf-8")

    with pytest.raises(ValueError, match="unknown teleop config profile key"):
        load_profile_overrides(path, "MuJoCoPickLift-v1", PickConfig())


def test_load_profile_rejects_unknown_section_override_key(tmp_path) -> None:
    path = tmp_path / "profile.json"
    path.write_text('{"pick":{"spawn_min_radious":0.1}}', encoding="utf-8")

    with pytest.raises(ValueError, match="unknown teleop config override key"):
        load_profile_overrides(path, "MuJoCoPickLift-v1", PickConfig())


def test_overrides_from_mapping_accepts_object_specs() -> None:
    overrides = overrides_from_mapping({"object_specs": ["cube:red"], "n_distractors": 0})

    assert overrides.object_specs == ("cube:red",)
    assert overrides.n_distractors == 0


def test_overrides_to_mapping_round_trips_through_from_mapping() -> None:
    import json

    overrides = TeleopConfigOverrides(
        objects=(CubeObject(color="blue"), YCBObject(model_id="011_banana")),
        n_distractors=2,
        ground_colors=("red", "green"),
        spawn_max_radius=0.3,
        reset_settle_frames=5,
    )

    mapping = overrides_to_mapping(overrides)

    json.dumps(mapping)
    assert overrides_to_mapping(overrides_from_mapping(mapping)) == mapping


def test_object_to_mapping_serializes_mesh() -> None:
    mesh = MeshObject(
        collision_mesh_path="c.obj",
        visual_mesh_path="v.obj",
        mass=0.2,
        name="widget",
        scale=2.0,
    )

    mapping = object_to_mapping(mesh)

    assert object_to_mapping(object_from_mapping(mapping)) == mapping


def test_overrides_from_mapping_rejects_non_finite_float() -> None:
    with pytest.raises(ValueError, match="finite"):
        overrides_from_mapping({"spawn_max_radius": "nan"})


def test_overrides_from_mapping_rejects_non_integer_reset_settle_frames() -> None:
    with pytest.raises(ValueError, match="reset_settle_frames"):
        overrides_from_mapping({"reset_settle_frames": 1.5})


def test_load_config_factory_wraps_missing_modules() -> None:
    with pytest.raises(ValueError, match="--env-config-factory module"):
        load_config_factory("missing_teleop_factory:build")


def test_load_config_factory_invokes_module_function(tmp_path, monkeypatch) -> None:
    module_path = tmp_path / "custom_factory.py"
    module_path.write_text(
        "from so101_nexus import PickConfig, CubeObject\n"
        "def build(env_id, base_config):\n"
        "    assert env_id == 'MuJoCoPickLift-v1'\n"
        "    return PickConfig(objects=[CubeObject(color='blue')])\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    factory = load_config_factory("custom_factory:build")
    update = apply_config_factory(factory, "MuJoCoPickLift-v1", PickConfig())
    cfg = update.config

    assert isinstance(cfg, PickConfig)
    assert isinstance(cfg.objects[0], CubeObject)
    assert cfg.objects[0].color == "blue"


def test_apply_config_factory_has_no_dead_kwargs_parameter() -> None:
    assert "kwargs" not in inspect.signature(apply_config_factory).parameters


def test_apply_config_factory_can_return_gym_kwargs(tmp_path, monkeypatch) -> None:
    module_path = tmp_path / "custom_kwargs.py"
    module_path.write_text(
        "from so101_nexus import PickConfig\n"
        "def build(env_id, base_config):\n"
        "    return {'config': PickConfig(n_distractors=0), 'custom_flag': True}\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    update = apply_config_factory(load_config_factory("custom_kwargs:build"), "x", PickConfig())
    cfg = update.config

    assert isinstance(cfg, PickConfig)
    assert cfg.n_distractors == 0
    assert update.kwargs == {"custom_flag": True}


def test_apply_config_factory_rejects_none_return(tmp_path, monkeypatch) -> None:
    module_path = tmp_path / "custom_none.py"
    module_path.write_text(
        "def build(env_id, base_config):\n    return None\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(ValueError, match="must return"):
        apply_config_factory(load_config_factory("custom_none:build"), "x", PickConfig())
