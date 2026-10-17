"""Distractor bounds share validation across task families and teleop profiles."""

import pytest

from so101_nexus import CubeObject, PickAndPlaceConfig, PickConfig, StackCubeConfig
from so101_nexus.teleop.config_customization import (
    apply_config_overrides,
    overrides_from_mapping,
    overrides_to_mapping,
)


@pytest.mark.parametrize("config_cls", [PickConfig, PickAndPlaceConfig, StackCubeConfig])
def test_distractor_bounds(config_cls):
    kwargs = (
        {"objects": [CubeObject(color=c) for c in ("red", "green", "blue")]}
        if config_cls is PickConfig
        else {}
    )
    cfg = config_cls(min_distractors=0, max_distractors=2, **kwargs)
    assert (cfg.min_distractors, cfg.max_distractors) == (0, 2)
    clone = config_cls(**vars(cfg))
    assert (clone.min_distractors, clone.max_distractors) == (0, 2)
    upper_only = config_cls(max_distractors=2, **kwargs)
    assert (upper_only.min_distractors, upper_only.max_distractors) == (0, 2)
    fixed = config_cls(n_distractors=2, **kwargs)
    assert (fixed.min_distractors, fixed.max_distractors) == (2, 2)


@pytest.mark.parametrize("bounds", [(-1, 2), (2, 1), (0, 4), (0.5, 1), (0, True)])
@pytest.mark.parametrize("config_cls", [PickConfig, PickAndPlaceConfig, StackCubeConfig])
def test_invalid_distractor_bounds(config_cls, bounds):
    with pytest.raises(ValueError, match="distractor"):
        config_cls(min_distractors=bounds[0], max_distractors=bounds[1])


def test_teleop_range_round_trip_and_fixed_override():
    raw = {"min_distractors": 0, "max_distractors": 2}
    overrides = overrides_from_mapping(raw)
    assert overrides_to_mapping(overrides) == raw
    cfg = apply_config_overrides(StackCubeConfig(n_distractors=1), overrides)
    assert (cfg.min_distractors, cfg.max_distractors) == (0, 2)
    fixed = apply_config_overrides(cfg, overrides_from_mapping({"n_distractors": 1}))
    assert (fixed.min_distractors, fixed.max_distractors) == (1, 1)


@pytest.mark.parametrize(
    "raw",
    [
        {"min_distractors": -1},
        {"max_distractors": 1.5},
        {"max_distractors": True},
        {"min_distractors": 2, "max_distractors": 1},
    ],
)
def test_teleop_rejects_invalid_bounds(raw):
    with pytest.raises(ValueError, match="distractor"):
        overrides_from_mapping(raw)
