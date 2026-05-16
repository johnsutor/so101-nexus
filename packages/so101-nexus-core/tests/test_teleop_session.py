"""Unit tests for so101_nexus_core.teleop.session pure helpers.

These tests cover ``_default_repo_id``, ``_resolve_env_config``,
``_replace_wrist_camera``, ``_replace_overhead_camera``, and
``_wire_camera_observations`` — all of which are pure-logic helpers
that don't require gymnasium or any optional dependency.
"""

import datetime
import re
from pathlib import Path

import numpy as np
import pytest

from so101_nexus_core.config import PickConfig, ReachConfig
from so101_nexus_core.objects import CubeObject, YCBObject
from so101_nexus_core.observations import OverheadCamera, WristCamera
from so101_nexus_core.teleop.config_customization import TeleopConfigOverrides
from so101_nexus_core.teleop.session import (
    _build_recording_config,
    _default_repo_id,
    _recording_env_kwargs,
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


def test_build_recording_config_applies_overrides_before_camera_wiring() -> None:
    cfg = _build_recording_config(
        PickConfig(),
        (320, 240),
        (640, 480),
        overrides=TeleopConfigOverrides(
            object_specs=("cube:green", "ycb:011_banana"),
            n_distractors=1,
        ),
        env_id="MuJoCoPickLift-v1",
    )

    assert cfg.n_distractors == 1
    assert isinstance(cfg.objects[0], CubeObject)
    assert cfg.objects[0].color == "green"
    assert isinstance(cfg.objects[1], YCBObject)
    assert cfg.objects[1].model_id == "011_banana"
    assert any(isinstance(o, WristCamera) for o in cfg.observations)
    assert any(isinstance(o, OverheadCamera) for o in cfg.observations)


def test_build_recording_config_applies_factory_after_profile(tmp_path: Path) -> None:
    profile = tmp_path / "profile.json"
    profile.write_text('{"pick":{"n_distractors":1,"object_specs":["cube:red","cube:blue"]}}')

    def _factory(_env_id: str, base_config: object | None) -> PickConfig:
        assert isinstance(base_config, PickConfig)
        assert base_config.n_distractors == 1
        return PickConfig(objects=[CubeObject(color="green")], n_distractors=0)

    cfg = _build_recording_config(
        PickConfig(),
        (320, 240),
        (640, 480),
        profile_path=str(profile),
        env_id="MuJoCoPickLift-v1",
        factory=_factory,
    )

    assert cfg.n_distractors == 0
    assert cfg.objects[0].color == "green"


def test_recording_env_kwargs_applies_overrides_to_registered_config_kwargs(monkeypatch) -> None:
    def _entry_point():
        raise AssertionError("not used")

    monkeypatch.setattr(
        "so101_nexus_core.teleop.session._resolve_env_ctor",
        lambda _env_id: (_entry_point, {"config": PickConfig()}),
    )

    kwargs = _recording_env_kwargs(
        "CustomPick-v1",
        (320, 240),
        (640, 480),
        overrides=TeleopConfigOverrides(
            object_specs=("cube:green", "ycb:011_banana"),
            n_distractors=1,
        ),
    )

    cfg = kwargs["config"]
    assert cfg.n_distractors == 1
    assert cfg.objects[0].color == "green"
    assert isinstance(cfg.objects[1], YCBObject)
    assert any(isinstance(o, WristCamera) for o in cfg.observations)
    assert any(isinstance(o, OverheadCamera) for o in cfg.observations)


def test_recording_env_kwargs_applies_overrides_to_factory_config_for_no_default_env(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "so101_nexus_core.teleop.session._resolve_env_ctor",
        lambda _env_id: (_NoDefaultConfigEnv, {}),
    )

    kwargs = _recording_env_kwargs(
        "CustomPick-v1",
        (320, 240),
        (640, 480),
        overrides=TeleopConfigOverrides(object_specs=("cube:blue",)),
        factory=lambda _env_id, _base_config: PickConfig(),
    )

    assert kwargs["config"].objects[0].color == "blue"


def test_recording_env_kwargs_rejects_overrides_without_config(monkeypatch) -> None:
    monkeypatch.setattr(
        "so101_nexus_core.teleop.session._resolve_env_ctor",
        lambda _env_id: (_NoDefaultConfigEnv, {}),
    )

    with pytest.raises(ValueError, match="requires a config object"):
        _recording_env_kwargs(
            "CustomPick-v1",
            (320, 240),
            (640, 480),
            overrides=TeleopConfigOverrides(object_specs=("cube:blue",)),
        )


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


def test_make_state_plot_labels_lerobot_units() -> None:
    pytest.importorskip("plotly")

    from so101_nexus_core.teleop.session import make_state_plot

    fig = make_state_plot([np.zeros(2, dtype=np.float32)], ("a", "b"), fps=30)

    assert fig.layout.yaxis.title.text == "Position (deg / RANGE_0_100)"


def test_prepare_follower_calibration_creates_file_when_missing(tmp_path: Path) -> None:
    from so101_nexus_core.teleop.session import prepare_follower_calibration

    calibration_dir = tmp_path / "calibration" / "robots" / "sim_so_follower"
    fpath = prepare_follower_calibration(
        calibration_dir=calibration_dir,
        robot_id="teleop_sim",
    )

    assert fpath.exists()
    assert fpath.name == "teleop_sim.json"


def test_prepare_follower_calibration_is_idempotent(tmp_path: Path) -> None:
    from so101_nexus_core.teleop.session import prepare_follower_calibration

    calibration_dir = tmp_path / "cal"
    fpath_a = prepare_follower_calibration(
        calibration_dir=calibration_dir,
        robot_id="teleop_sim",
    )
    mtime_a = fpath_a.stat().st_mtime_ns
    fpath_b = prepare_follower_calibration(
        calibration_dir=calibration_dir,
        robot_id="teleop_sim",
    )

    assert fpath_a == fpath_b
    assert fpath_b.stat().st_mtime_ns == mtime_a


def test_build_sim_follower_config_wires_cameras_and_env_kwargs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from so101_nexus_core.lerobot_adapter.sim_camera_config import SimCameraConfig
    from so101_nexus_core.lerobot_adapter.sim_follower_config import SimSOFollowerConfig
    from so101_nexus_core.teleop import session as teleop_session
    from so101_nexus_core.teleop.session import build_sim_follower_config

    seen: dict[str, object] = {}

    def _fake_recording_env_kwargs(
        env_id,
        wrist_wh,
        overhead_wh,
        *,
        overrides=None,
        profile_path=None,
        factory=None,
    ):
        seen.update(
            env_id=env_id,
            wrist_wh=wrist_wh,
            overhead_wh=overhead_wh,
            overrides=overrides,
            profile_path=profile_path,
            factory=factory,
        )
        return {"custom": "kept"}

    monkeypatch.setattr(
        teleop_session,
        "_recording_env_kwargs",
        _fake_recording_env_kwargs,
    )

    config = build_sim_follower_config(
        env_id="MuJoCoReach-v1",
        robot_id="teleop_sim",
        wrist_wh=(320, 240),
        overhead_wh=(640, 360),
        fps=15,
        calibration_dir=tmp_path,
        profile_path="profile.toml",
    )

    assert isinstance(config, SimSOFollowerConfig)
    assert config.env_id == "MuJoCoReach-v1"
    assert config.env_kwargs == {"custom": "kept"}
    assert config.use_degrees is True
    assert config.id == "teleop_sim"
    assert config.calibration_dir == tmp_path
    assert seen["profile_path"] == "profile.toml"
    assert set(config.cameras) == {"wrist", "overhead"}
    wrist = config.cameras["wrist"]
    overhead = config.cameras["overhead"]
    assert isinstance(wrist, SimCameraConfig)
    assert wrist.width == 320
    assert wrist.height == 240
    assert wrist.fps == 15
    assert wrist.source == "wrist_camera"
    assert isinstance(overhead, SimCameraConfig)
    assert overhead.width == 640
    assert overhead.height == 360
    assert overhead.fps == 15
    assert overhead.source == "overhead_camera"
