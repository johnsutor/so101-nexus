"""Tests for asset-path accessors."""

from so101_nexus_core import (
    get_so101_mujoco_model_dir,
    get_so101_mujoco_model_path,
)


def test_mujoco_model_path_points_at_menagerie():
    d = get_so101_mujoco_model_dir()
    p = get_so101_mujoco_model_path()
    assert d.name == "SO101_menagerie"
    assert p == d / "so101.xml"
    assert p.is_file()
    assert (d / "assets").is_dir()
