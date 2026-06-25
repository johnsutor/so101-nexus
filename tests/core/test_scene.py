"""Shared scene builder and option presets."""

import tempfile

import mujoco

from so101_nexus import get_so101_mujoco_model_dir, get_so101_mujoco_model_path
from so101_nexus.scene import (
    MUJOCO_SCENE_OPTION_XML,
    WARP_SCENE_OPTION_XML,
)


def test_mujoco_option_unchanged():
    assert 'integrator="implicitfast"' in MUJOCO_SCENE_OPTION_XML
    assert "noslip_iterations" in MUJOCO_SCENE_OPTION_XML


def test_warp_option_drops_unsupported_features():
    assert "implicitfast" not in WARP_SCENE_OPTION_XML
    assert "noslip" not in WARP_SCENE_OPTION_XML
    assert 'integrator="implicit"' in WARP_SCENE_OPTION_XML
    for token in (
        'timestep="0.005"',
        'cone="elliptic"',
        'impratio="10"',
        'iterations="10"',
        'ls_iterations="20"',
    ):
        assert token in WARP_SCENE_OPTION_XML


def test_robot_floor_builder_compiles_with_both_options():
    from so101_nexus.scene import build_robot_floor_scene_xml

    for option in (MUJOCO_SCENE_OPTION_XML, WARP_SCENE_OPTION_XML):
        xml = build_robot_floor_scene_xml(
            [0.5, 0.5, 0.5, 1.0],
            option_xml=option,
            robot_xml_path=str(get_so101_mujoco_model_path()),
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".xml", dir=get_so101_mujoco_model_dir(), delete=True
        ) as f:
            f.write(xml)
            f.flush()
            model = mujoco.MjModel.from_xml_path(f.name)
        assert model.nq > 0
        # No target site/body: robot + floor only.
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "reach_target") == -1


def _compile_scene(xml: str) -> mujoco.MjModel:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".xml", dir=get_so101_mujoco_model_dir(), delete=True
    ) as f:
        f.write(xml)
        f.flush()
        return mujoco.MjModel.from_xml_path(f.name)


def test_scenes_have_single_shadow_caster_and_antialiasing():
    """Built scenes cast exactly one shadow and apply high-quality AA settings.

    Guards the rendering fixes: two shadow-casting lights produced a doubled
    robot shadow, and the MuJoCo default offscreen quality (offsamples=4,
    shadowsize=4096, shadowclip=1.0) left blocky shadow and silhouette aliasing
    in camera observations.
    """
    from so101_nexus.object_slots import build_object_scene_xml
    from so101_nexus.objects import CubeObject
    from so101_nexus.scene import build_robot_floor_scene_xml

    robot_path = str(get_so101_mujoco_model_path())
    scenes = [
        build_robot_floor_scene_xml(
            [0.5, 0.5, 0.5, 1.0],
            option_xml=MUJOCO_SCENE_OPTION_XML,
            robot_xml_path=robot_path,
        ),
        build_object_scene_xml(
            [CubeObject()],
            ["pick_slot_0"],
            [0.5, 0.5, 0.5, 1.0],
            option_xml=MUJOCO_SCENE_OPTION_XML,
            robot_xml_path=robot_path,
        ),
    ]
    for xml in scenes:
        model = _compile_scene(xml)
        assert model.nlight == 2
        assert int(model.light_castshadow.sum()) == 1
        assert model.vis.quality.offsamples >= 8
        assert model.vis.quality.shadowsize >= 8192
        assert model.vis.map.shadowclip <= 0.5
