"""Unit tests for PrimitiveTargetSpec components (no backend deps)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from so101_nexus_core.config import LookAtConfig, MoveConfig, ReachConfig
from so101_nexus_core.tasks import (
    NumpyContext,
    make_look_at_spec,
    make_move_spec,
    make_reach_spec,
    resolve_task_description,
)

if TYPE_CHECKING:
    from so101_nexus_core.objects import CubeObject


def test_reach_spec_sampler_returns_3d_position_above_floor():
    cfg = ReachConfig()
    spec = make_reach_spec(cfg)
    rng = np.random.default_rng(42)
    pos = spec.sampler.sample_numpy(NumpyContext(rng=rng, config=cfg))
    assert pos.shape == (3,)
    assert pos[2] >= cfg.target_radius


def test_reach_sampler_is_seeded_deterministic():
    cfg = ReachConfig()
    spec = make_reach_spec(cfg)
    a = spec.sampler.sample_numpy(NumpyContext(rng=np.random.default_rng(0), config=cfg))
    b = spec.sampler.sample_numpy(NumpyContext(rng=np.random.default_rng(0), config=cfg))
    np.testing.assert_array_equal(a, b)


def test_move_sampler_requires_tcp_pos():
    cfg = MoveConfig()
    spec = make_move_spec(cfg)
    rng = np.random.default_rng(0)
    with pytest.raises(AssertionError):
        spec.sampler.sample_numpy(NumpyContext(rng=rng, config=cfg, tcp_pos=None))


def test_move_sampler_translates_tcp_by_direction_distance():
    cfg = MoveConfig(direction="up", target_distance=0.10)
    spec = make_move_spec(cfg)
    tcp = np.array([0.10, 0.0, 0.05])
    pos = spec.sampler.sample_numpy(
        NumpyContext(rng=np.random.default_rng(0), config=cfg, tcp_pos=tcp)
    )
    np.testing.assert_allclose(pos, np.array([0.10, 0.0, 0.15]), atol=1e-6)


def test_look_at_sampler_returns_floor_position():
    cfg = LookAtConfig()
    spec = make_look_at_spec(cfg)
    rng = np.random.default_rng(7)
    pos = spec.sampler.sample_numpy(NumpyContext(rng=rng, config=cfg))
    assert pos.shape == (3,)
    target_obj: CubeObject = cfg.objects[0]
    np.testing.assert_allclose(pos[2], target_obj.half_size, atol=1e-9)


def test_distance_metric_success_below_threshold():
    cfg = ReachConfig(success_threshold=0.05)
    spec = make_reach_spec(cfg)
    target = np.array([0.1, 0.0, 0.1])
    near = np.array([0.11, 0.0, 0.1])
    far = np.array([0.5, 0.0, 0.1])
    metric_near, ok_near = spec.metric.evaluate_numpy(
        target_pos=target,
        tcp_pos=near,
        tcp_forward=None,
        ctx=NumpyContext(rng=np.random.default_rng(0), config=cfg),
    )
    metric_far, ok_far = spec.metric.evaluate_numpy(
        target_pos=target,
        tcp_pos=far,
        tcp_forward=None,
        ctx=NumpyContext(rng=np.random.default_rng(0), config=cfg),
    )
    assert ok_near is True
    assert ok_far is False
    assert metric_near < metric_far


def test_tanh_shaper_in_unit_interval():
    cfg = ReachConfig()
    spec = make_reach_spec(cfg)
    ctx = NumpyContext(rng=np.random.default_rng(0), config=cfg)
    for d in [0.0, 0.05, 0.5, 5.0]:
        v = spec.shaper.shape_numpy(d, ctx)
        assert 0.0 <= v <= 1.0


def test_resolve_task_description_static_string():
    cfg = ReachConfig()
    spec = make_reach_spec(cfg)
    desc = resolve_task_description(spec, cfg)
    assert isinstance(desc, str)
    assert desc


def test_resolve_task_description_callable_renders_with_config():
    cfg = MoveConfig(direction="left", target_distance=0.07)
    spec = make_move_spec(cfg)
    desc = resolve_task_description(spec, cfg)
    assert "left" in desc
    assert "0.07" in desc
