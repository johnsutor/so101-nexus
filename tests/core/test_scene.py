"""Shared scene builder and option presets."""

import tempfile

import mujoco

from so101_nexus import get_so101_mujoco_model_dir, get_so101_mujoco_model_path
from so101_nexus.scene import (
    MUJOCO_SCENE_OPTION_XML,
    WARP_SCENE_OPTION_XML,
    build_reach_scene_xml,
)


def test_mujoco_option_unchanged():
    assert 'integrator="implicitfast"' in MUJOCO_SCENE_OPTION_XML
    assert "noslip_iterations" in MUJOCO_SCENE_OPTION_XML


def test_warp_option_drops_unsupported_features():
    assert "implicitfast" not in WARP_SCENE_OPTION_XML
    assert "noslip" not in WARP_SCENE_OPTION_XML
    assert 'integrator="implicit"' in WARP_SCENE_OPTION_XML
    for token in ('timestep="0.005"', 'cone="elliptic"', 'impratio="10"',
                  'iterations="10"', 'ls_iterations="20"'):
        assert token in WARP_SCENE_OPTION_XML


def test_builder_compiles_with_both_options():
    for option in (MUJOCO_SCENE_OPTION_XML, WARP_SCENE_OPTION_XML):
        xml = build_reach_scene_xml(
            [0.5, 0.5, 0.5, 1.0],
            0.02,
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
