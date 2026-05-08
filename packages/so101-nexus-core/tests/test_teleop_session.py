"""Unit tests for so101_nexus_core.teleop.session pure helpers.

These tests cover ``_default_repo_id``, ``_resolve_env_config``,
``_replace_wrist_camera``, ``_replace_overhead_camera``, and
``_wire_camera_observations`` — all of which are pure-logic helpers
that don't require gymnasium or any optional dependency.
"""

import datetime
import re

import numpy as np
import pytest

from so101_nexus_core.config import ReachConfig
from so101_nexus_core.observations import OverheadCamera, WristCamera
from so101_nexus_core.teleop.session import (
    _default_repo_id,
    _replace_overhead_camera,
    _replace_wrist_camera,
    _resolve_env_config,
    _wire_camera_observations,
)

# ---------------------------------------------------------------------------
# _default_repo_id
# ---------------------------------------------------------------------------


def test_default_repo_id_format(monkeypatch) -> None:
    """Repo id encodes env_id and a 15-char timestamp."""
    fixed = datetime.datetime(2026, 4, 18, 9, 30, 45)

    class _FixedDatetime(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed

    monkeypatch.setattr("so101_nexus_core.teleop.session.datetime.datetime", _FixedDatetime)

    repo_id = _default_repo_id("Reach-v0")

    assert repo_id == "local/teleop-Reach-v0-20260418_093045"


def test_default_repo_id_sanitizes_slashes_and_spaces() -> None:
    """Slashes become dashes; spaces become underscores."""
    repo_id = _default_repo_id("foo/bar baz")

    slug_match = re.match(r"local/teleop-(.+)-\d{8}_\d{6}$", repo_id)
    assert slug_match is not None, repo_id
    assert slug_match.group(1) == "foo-bar_baz"


# ---------------------------------------------------------------------------
# _resolve_env_config
# ---------------------------------------------------------------------------


class _ExplicitConfigEnv:
    default_config_cls = ReachConfig


def test_resolve_env_config_uses_default_config_cls() -> None:
    result = _resolve_env_config(_ExplicitConfigEnv)

    assert isinstance(result, ReachConfig)


class _NoDefaultConfigEnv:
    pass


def test_resolve_env_config_returns_none_without_default_config_cls() -> None:
    assert _resolve_env_config(_NoDefaultConfigEnv) is None


def test_resolve_env_config_propagates_config_construction_errors() -> None:
    class _RaisingConfig:
        def __init__(self) -> None:
            raise RuntimeError("boom")

    class _RaisingConfigEnv:
        default_config_cls = _RaisingConfig

    with pytest.raises(RuntimeError, match="boom"):
        _resolve_env_config(_RaisingConfigEnv)


# ---------------------------------------------------------------------------
# Camera-replacement helpers
# ---------------------------------------------------------------------------


def test_replace_wrist_camera_preserves_other_fields() -> None:
    original = WristCamera(width=160, height=120, pos_x_noise=0.05, pos_y_noise=0.07)

    replaced = _replace_wrist_camera(original, width=320, height=240)

    assert replaced.width == 320
    assert replaced.height == 240
    assert replaced.pos_x_noise == 0.05
    assert replaced.pos_y_noise == 0.07
    assert replaced.fov_deg_range == original.fov_deg_range


def test_replace_overhead_camera_preserves_fov() -> None:
    original = OverheadCamera(width=160, height=120, fov_deg=42.0)

    replaced = _replace_overhead_camera(original, width=320, height=240)

    assert replaced.width == 320
    assert replaced.height == 240
    assert replaced.fov_deg == 42.0


# ---------------------------------------------------------------------------
# _wire_camera_observations
# ---------------------------------------------------------------------------


def test_wire_camera_observations_resizes_existing_cameras() -> None:
    obs = [WristCamera(width=160, height=120), OverheadCamera(width=160, height=120)]

    out = _wire_camera_observations(obs, wrist_wh=(320, 240), overhead_wh=(640, 480))

    wrist = next(c for c in out if isinstance(c, WristCamera))
    overhead = next(c for c in out if isinstance(c, OverheadCamera))
    assert (wrist.width, wrist.height) == (320, 240)
    assert (overhead.width, overhead.height) == (640, 480)


def test_wire_camera_observations_appends_missing_cameras() -> None:
    """If the env's observations list lacks one or both cameras, defaults are appended."""

    class _OtherObservation:
        pass

    obs = [_OtherObservation()]

    out = _wire_camera_observations(obs, wrist_wh=(320, 240), overhead_wh=(640, 480))

    assert any(isinstance(c, WristCamera) and c.width == 320 for c in out)
    assert any(isinstance(c, OverheadCamera) and c.width == 640 for c in out)
    assert any(isinstance(c, _OtherObservation) for c in out)


# ---------------------------------------------------------------------------
# Optional: state-plot rendering (gated on plotly)
# ---------------------------------------------------------------------------


def test_make_state_plot_returns_figure_with_one_trace_per_joint() -> None:
    pytest.importorskip("plotly")

    from so101_nexus_core.teleop.session import make_state_plot

    states = [np.array([0.0, 0.1, 0.2]), np.array([0.5, 0.4, 0.3]), np.array([1.0, 0.0, -0.5])]
    joint_names = ("a", "b", "c")

    fig = make_state_plot(states, joint_names, fps=30)

    assert len(fig.data) == 3
    assert {trace.name for trace in fig.data} == {"a", "b", "c"}
