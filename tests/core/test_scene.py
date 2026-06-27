"""Shared scene builder and option presets."""

import tempfile

import mujoco
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

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


_unit = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False).map(
    lambda v: round(v, 4)
)
_ground_rgba = st.tuples(
    _unit,
    _unit,
    _unit,
    st.floats(min_value=0.1, max_value=1.0, allow_nan=False).map(lambda v: round(v, 4)),
).map(list)


@given(
    rgba=_ground_rgba,
    with_object=st.booleans(),
    half_size=st.floats(min_value=0.01, max_value=0.04, allow_nan=False),
)
@settings(max_examples=25, deadline=None)
def test_scenes_use_spotlight_shadow_caster_and_edge_antialiasing(rgba, with_object, half_size):
    """Every built scene casts exactly one spotlight shadow and keeps edge MSAA.

    Property: regardless of ground colour or scene contents, the AA/shadow contract
    holds. It guards the rendering fix: a *directional* light must shadow-map the
    entire infinite floor, so its far grazing texels dither into a speckle moire on
    the ground in wrist-camera views ("weird textures in the ground"). MSAA cannot
    remove it (a per-fragment depth-test artifact, not an edge) and raising
    ``shadowsize`` worsened it. The shadow caster is therefore a spotlight, whose
    bounded cone frustum concentrates shadow texels on the workspace; the overhead
    fill stays directional and non-casting (a single caster avoids a doubled
    shadow). ``offsamples`` stays above the MuJoCo default (4) for edge AA.
    """
    from so101_nexus.object_slots import build_object_scene_xml
    from so101_nexus.objects import CubeObject
    from so101_nexus.scene import build_robot_floor_scene_xml

    robot_path = str(get_so101_mujoco_model_path())
    if with_object:
        xml = build_object_scene_xml(
            [CubeObject(half_size=half_size)],
            ["pick_slot_0"],
            rgba,
            option_xml=MUJOCO_SCENE_OPTION_XML,
            robot_xml_path=robot_path,
        )
    else:
        xml = build_robot_floor_scene_xml(
            rgba, option_xml=MUJOCO_SCENE_OPTION_XML, robot_xml_path=robot_path
        )

    model = _compile_scene(xml)
    assert model.nlight == 2
    casters = np.flatnonzero(model.light_castshadow)
    assert casters.size == 1, "exactly one shadow caster (a second caster doubles the robot shadow)"
    assert model.light_type[casters[0]] == int(mujoco.mjtLightType.mjLIGHT_SPOT)
    assert model.vis.quality.offsamples >= 8
    assert model.vis.quality.shadowsize >= 8192
