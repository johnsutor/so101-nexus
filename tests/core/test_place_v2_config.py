"""Versioned placement configuration and evaluator metadata contracts."""

import json

import pytest

from so101_nexus import PickAndPlaceConfig, PickAndPlaceV2Config


def test_precision_region_has_one_radius_and_round_trips():
    config = PickAndPlaceV2Config(target_disc_radius=0.03, placement_dwell_time=0.3)
    restored = PickAndPlaceV2Config(**vars(config))
    assert restored.goal_thresh == restored.target_disc_radius == 0.03
    contract = json.loads(json.dumps(restored.placement_contract))
    assert contract["target_radius"] == 0.03
    assert contract["placement_dwell_time"] == 0.3
    assert contract["placement_mode"] == "center"
    assert PickAndPlaceConfig().goal_thresh == 0.025


@pytest.mark.parametrize("mode", ["overlap", "containment"])
def test_other_task_definitions_are_not_silent_center_aliases(mode):
    with pytest.raises(ValueError, match="placement_mode"):
        PickAndPlaceV2Config(placement_mode=mode)


def test_invisible_precision_region_is_rejected():
    with pytest.raises(ValueError, match="goal_thresh"):
        PickAndPlaceV2Config(target_disc_radius=0.05, goal_thresh=0.025)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_disc_radius": float("nan")},
        {"placement_dwell_time": 0},
        {"placement_dwell_time": float("inf")},
        {"support_min_weight_fraction": 0},
        {"support_min_weight_fraction": 1.1},
        {"support_force_tolerance": -1},
        {"support_relative_force_tolerance": float("nan")},
        {"footprint_scanlines": 0},
        {"footprint_scanlines": 10.5},
        {"object_static_lin_threshold": float("nan")},
    ],
)
def test_invalid_evaluator_thresholds_fail_before_construction(kwargs):
    with pytest.raises(ValueError, match="must"):
        PickAndPlaceV2Config(**kwargs)
