"""Unit tests for so101_nexus_core.teleop.session pure helpers.

These tests cover ``_default_repo_id``, ``_resolve_env_config``,
``_replace_wrist_camera``, ``_replace_overhead_camera``, and
``_wire_camera_observations`` — all of which are pure-logic helpers
that don't require gymnasium or any optional dependency.

Note: this module deliberately does NOT use ``from __future__ import annotations``
because some tests rely on the runtime form of class annotations being a real
type vs. a string.
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


class _NoAnnotationEnv:
    def __init__(self, config):
        self.config = config


def test_resolve_env_config_returns_none_when_no_string_annotation() -> None:
    """No `config` class annotation → config_class_name stays None → returns None."""
    assert _resolve_env_config(_NoAnnotationEnv) is None


class _EnvWithDefaultConfig:
    default_config_cls = ReachConfig

    def __init__(self, config=None):
        self.config = config


class _EnvWithoutDefaultConfig:
    def __init__(self, config=None):
        self.config = config


class _RaisingConfig:
    def __init__(self):
        raise RuntimeError("boom")


class _EnvWithRaisingConfig:
    default_config_cls = _RaisingConfig

    def __init__(self, config=None):
        self.config = config


def test_resolve_env_config_uses_default_config_cls() -> None:
    """Env class with default_config_cls returns a config instance."""
    result = _resolve_env_config(_EnvWithDefaultConfig)
    assert isinstance(result, ReachConfig)


def test_resolve_env_config_returns_none_without_attribute() -> None:
    """Env class without default_config_cls returns None."""
    assert _resolve_env_config(_EnvWithoutDefaultConfig) is None


def test_resolve_env_config_propagates_init_errors() -> None:
    """A default_config_cls whose __init__ raises propagates the error.

    Behavior change vs the old reflection path which silently swallowed
    the exception: a real env config that cannot construct is a bug
    worth surfacing.
    """
    import pytest

    with pytest.raises(RuntimeError, match="boom"):
        _resolve_env_config(_EnvWithRaisingConfig)


def test_every_real_env_class_declares_default_config_cls() -> None:
    """Regression guard: every env class on every backend has the attribute."""
    import importlib

    cases = [
        ("so101_nexus_mujoco.reach_env", "ReachEnv"),
        ("so101_nexus_mujoco.move_env", "MoveEnv"),
        ("so101_nexus_mujoco.look_at_env", "LookAtEnv"),
        ("so101_nexus_mujoco.pick_env", "PickEnv"),
        ("so101_nexus_mujoco.pick_env", "PickLiftEnv"),
        ("so101_nexus_mujoco.pick_and_place", "PickAndPlaceEnv"),
        ("so101_nexus_maniskill.reach_env", "ReachEnv"),
        ("so101_nexus_maniskill.move_env", "MoveEnv"),
        ("so101_nexus_maniskill.look_at_env", "LookAtEnv"),
        ("so101_nexus_maniskill.pick_env", "PickEnv"),
        ("so101_nexus_maniskill.pick_env", "PickLiftEnv"),
        ("so101_nexus_maniskill.pick_and_place", "PickAndPlaceEnv"),
    ]
    from so101_nexus_core.config import EnvironmentConfig

    for module_name, class_name in cases:
        try:
            mod = importlib.import_module(module_name)
        except ImportError:
            # Backend not installed in this test environment; skip.
            continue
        cls = getattr(mod, class_name)
        attr = getattr(cls, "default_config_cls", None)
        assert attr is not None, f"{class_name} missing default_config_cls"
        assert issubclass(attr, EnvironmentConfig), (
            f"{class_name}.default_config_cls is not an EnvironmentConfig subclass"
        )


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
