from pathlib import Path

import pytest

from so101_nexus_core.types import YCB_OBJECTS
from so101_nexus_core.ycb_assets import get_ycb_mesh_dir

EXPECTED_MODEL_IDS = [
    "009_gelatin_box",
    "011_banana",
    "030_fork",
    "031_spoon",
    "032_knife",
    "033_spatula",
    "037_scissors",
    "040_large_marker",
    "043_phillips_screwdriver",
    "058_golf_ball",
]


class TestYCBConstants:
    def test_ycb_objects_has_10_entries(self):
        assert len(YCB_OBJECTS) == 10

    def test_ycb_objects_contains_expected_ids(self):
        assert set(YCB_OBJECTS.keys()) == set(EXPECTED_MODEL_IDS)

    def test_ycb_objects_values_are_strings(self):
        for model_id, name in YCB_OBJECTS.items():
            assert isinstance(name, str), f"{model_id} name is not a string"
            assert len(name) > 0, f"{model_id} name is empty"


class TestYCBAssets:
    def test_get_ycb_mesh_dir_returns_path(self):
        result = get_ycb_mesh_dir("009_gelatin_box")
        assert isinstance(result, Path)

    def test_get_ycb_mesh_dir_invalid_model_raises(self):
        with pytest.raises(ValueError, match="model_id"):
            get_ycb_mesh_dir("invalid_model")

    def test_get_ycb_mesh_dir_contains_model_id(self):
        for model_id in EXPECTED_MODEL_IDS:
            path = get_ycb_mesh_dir(model_id)
            assert model_id in str(path)
