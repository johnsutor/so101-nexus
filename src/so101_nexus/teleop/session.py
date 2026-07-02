"""Env-kwarg resolution and review utilities for teleop sessions.

``_recording_env_kwargs`` walks the resolved env config's ``observations``
list and ensures both a :class:`WristCamera` and an :class:`OverheadCamera`
are present, sized to the requested resolutions. Existing camera instances
have their domain-randomisation parameters preserved; missing cameras get
appended with default parameters.
"""

from __future__ import annotations

import datetime
import importlib
import tempfile
from enum import Enum

import numpy as np

from so101_nexus.observations import OverheadCamera, WristCamera
from so101_nexus.teleop.config_customization import (
    ConfigFactory,
    ConfigFactoryUpdate,
    TeleopConfigOverrides,
    apply_config_factory,
    apply_config_overrides,
    load_profile_overrides,
)


class RepoIdStatus(str, Enum):  # noqa: UP042 - StrEnum requires Python 3.11.
    """Outcome of validating a HuggingFace dataset repo ID for teleop push."""

    OK = "ok"
    LOCAL_ONLY = "local_only"
    MISSING_NAMESPACE = "missing_namespace"
    INVALID_CHARS = "invalid_chars"


def validate_hub_repo_id(value: str) -> RepoIdStatus:
    """Classify a user-entered repo ID for Hub push readiness.

    Empty strings are local-only. Hub pushes require exactly one slash with
    non-empty namespace and dataset parts. Character validation is delegated
    to ``huggingface_hub`` so the accepted alphabet matches the Hub.
    """
    stripped = value.strip()
    if not stripped:
        return RepoIdStatus.LOCAL_ONLY

    parts = stripped.split("/")
    if len(parts) != 2 or not all(parts):
        return RepoIdStatus.MISSING_NAMESPACE

    from huggingface_hub.utils import HFValidationError, validate_repo_id

    try:
        validate_repo_id(stripped)
    except HFValidationError:
        return RepoIdStatus.INVALID_CHARS
    return RepoIdStatus.OK


def _default_repo_id(env_id: str) -> str:
    """Generate a local-only dataset repo ID from *env_id* and the current timestamp."""
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = env_id.replace("/", "-").replace(" ", "_")
    return f"local/teleop-{safe}-{ts}"


def _resolve_env_config(env_ctor: type) -> object | None:
    """Resolve the default config object for an environment class, or ``None``."""
    config_cls = getattr(env_ctor, "default_config_cls", None)
    if config_cls is None:
        return None
    return config_cls()


def _replace_wrist_camera(existing: WristCamera, width: int, height: int) -> WristCamera:
    """Return a new :class:`WristCamera` with the requested size, preserving other fields."""
    return WristCamera(
        width=width,
        height=height,
        fov_deg_range=existing.fov_deg_range,
        pitch_deg_range=existing.pitch_deg_range,
        pos_x_noise=existing.pos_x_noise,
        pos_y_center=existing.pos_y_center,
        pos_y_noise=existing.pos_y_noise,
        pos_z_center=existing.pos_z_center,
        pos_z_noise=existing.pos_z_noise,
    )


def _replace_overhead_camera(existing: OverheadCamera, width: int, height: int) -> OverheadCamera:
    """Return a new :class:`OverheadCamera` with the requested size, preserving ``fov_deg``."""
    return OverheadCamera(width=width, height=height, fov_deg=existing.fov_deg)


def _resolve_env_ctor(env_id: str):
    """Resolve the env constructor (class or callable) from a registered env id."""
    import gymnasium as gym

    from so101_nexus.teleop.leader import import_backend_for_env_id

    import_backend_for_env_id(env_id)
    spec = gym.spec(env_id)
    kwargs = dict(spec.kwargs)
    entry_point = spec.entry_point
    if isinstance(entry_point, str):
        module_name, attr_name = entry_point.split(":")
        env_ctor = getattr(importlib.import_module(module_name), attr_name)
    else:
        env_ctor = entry_point
    return env_ctor, kwargs


def _wire_camera_observations(
    observations: list,
    wrist_wh: tuple[int, int],
    overhead_wh: tuple[int, int],
) -> list:
    """Return *observations* with both cameras present and sized."""
    wrist_w, wrist_h = wrist_wh
    over_w, over_h = overhead_wh
    updated_obs: list = []
    found_wrist = False
    found_overhead = False
    for comp in observations:
        if isinstance(comp, WristCamera):
            updated_obs.append(_replace_wrist_camera(comp, wrist_w, wrist_h))
            found_wrist = True
        elif isinstance(comp, OverheadCamera):
            updated_obs.append(_replace_overhead_camera(comp, over_w, over_h))
            found_overhead = True
        else:
            updated_obs.append(comp)
    if not found_wrist:
        updated_obs.append(WristCamera(width=wrist_w, height=wrist_h))
    if not found_overhead:
        updated_obs.append(OverheadCamera(width=over_w, height=over_h))
    return updated_obs


def _recording_env_kwargs(
    env_id: str,
    wrist_wh: tuple[int, int],
    overhead_wh: tuple[int, int],
    *,
    overrides: TeleopConfigOverrides | None = None,
    profile_path: str | None = None,
    factory: ConfigFactory | None = None,
) -> dict:
    """Return ``gym.make`` kwargs for teleop recording with both cameras sized."""
    env_ctor, kwargs = _resolve_env_ctor(env_id)
    base_config = _resolve_env_config(env_ctor) if isinstance(env_ctor, type) else None
    _apply_recording_config_kwargs(
        kwargs,
        base_config=base_config,
        env_id=env_id,
        wrist_wh=wrist_wh,
        overhead_wh=overhead_wh,
        overrides=overrides,
        profile_path=profile_path,
        factory=factory,
    )
    return kwargs


def resolve_recording_observations(
    env_id: str,
    wrist_wh: tuple[int, int],
    overhead_wh: tuple[int, int],
    *,
    overrides: TeleopConfigOverrides | None = None,
    profile_path: str | None = None,
    factory: ConfigFactory | None = None,
) -> list | None:
    """Return the resolved recording config's ``observations`` list, or ``None``.

    Resolves the exact config the recording follower will run
    (``_recording_env_kwargs`` path), so the privileged-state schema derived from
    it matches the runtime ``observation.environment_state`` vector. ``None`` when
    the env exposes no config object (no privileged state to declare).
    """
    kwargs = _recording_env_kwargs(
        env_id,
        wrist_wh,
        overhead_wh,
        overrides=overrides,
        profile_path=profile_path,
        factory=factory,
    )
    config = kwargs.get("config")
    if config is None:
        return None
    return getattr(config, "observations", None)


def prepare_follower_calibration(*, calibration_dir, robot_id: str):
    """Ensure a SimSOFollower-compatible calibration file exists."""
    from pathlib import Path

    from so101_nexus.lerobot_adapter.synthetic_calibration import (
        write_synthetic_calibration,
    )

    calibration_dir = Path(calibration_dir)
    fpath = calibration_dir / f"{robot_id}.json"
    if fpath.exists():
        return fpath
    return write_synthetic_calibration(calibration_dir, robot_id)


def build_sim_follower_config(
    *,
    env_id: str,
    robot_id: str,
    wrist_wh: tuple[int, int],
    overhead_wh: tuple[int, int],
    fps: int = 30,
    calibration_dir=None,
    overrides: TeleopConfigOverrides | None = None,
    profile_path: str | None = None,
    factory: ConfigFactory | None = None,
):
    """Return a SimSOFollowerConfig matching the Gradio recording env choices."""
    from pathlib import Path

    from so101_nexus.lerobot_adapter.sim_camera_config import SimCameraConfig
    from so101_nexus.lerobot_adapter.sim_follower_config import SimSOFollowerConfig

    env_kwargs = _recording_env_kwargs(
        env_id,
        wrist_wh,
        overhead_wh,
        overrides=overrides,
        profile_path=profile_path,
        factory=factory,
    )
    wrist_w, wrist_h = wrist_wh
    overhead_w, overhead_h = overhead_wh
    cameras = {
        "wrist": SimCameraConfig(
            source="wrist_camera",
            width=wrist_w,
            height=wrist_h,
            fps=fps,
        ),
        "overhead": SimCameraConfig(
            source="overhead_camera",
            width=overhead_w,
            height=overhead_h,
            fps=fps,
        ),
    }
    return SimSOFollowerConfig(
        id=robot_id,
        calibration_dir=Path(calibration_dir) if calibration_dir is not None else None,
        env_id=env_id,
        env_kwargs=env_kwargs,
        cameras=cameras,
        use_degrees=True,
    )


def _apply_recording_config_kwargs(
    kwargs: dict,
    *,
    base_config: object | None,
    env_id: str,
    wrist_wh: tuple[int, int],
    overhead_wh: tuple[int, int],
    overrides: TeleopConfigOverrides | None = None,
    profile_path: str | None = None,
    factory: ConfigFactory | None = None,
) -> None:
    """Write recording config and factory kwargs into ``kwargs`` in one place."""
    config = base_config if base_config is not None else kwargs.get("config")
    if config is None:
        factory_update = apply_config_factory(factory, env_id, None)
        kwargs.update(factory_update.kwargs)
        if factory_update.config is None:
            if overrides is not None or profile_path is not None:
                raise ValueError(
                    "Teleop environment customization requires a config object. "
                    "Use an environment with default_config_cls, register a config "
                    "in Gymnasium kwargs, or provide --env-config-factory."
                )
            return
        config = _build_recording_config(
            factory_update.config,
            wrist_wh,
            overhead_wh,
            overrides=overrides,
            profile_path=profile_path,
            env_id=env_id,
        )
        kwargs["config"] = config
        return

    update = _customize_recording_config(
        config,
        wrist_wh,
        overhead_wh,
        overrides=overrides,
        profile_path=profile_path,
        env_id=env_id,
        factory=factory,
    )
    kwargs.update(update.kwargs)
    kwargs["config"] = update.config


def _build_recording_config(
    base_config: object,
    wrist_wh: tuple[int, int],
    overhead_wh: tuple[int, int],
    *,
    overrides: TeleopConfigOverrides | None = None,
    profile_path: str | None = None,
    env_id: str | None = None,
    factory: ConfigFactory | None = None,
) -> object:
    """Return a customized recording config with teleop cameras wired last."""
    update = _customize_recording_config(
        base_config,
        wrist_wh,
        overhead_wh,
        overrides=overrides,
        profile_path=profile_path,
        env_id=env_id,
        factory=factory,
    )
    if update.config is None:
        raise ValueError("recording config factory returned no config")
    return update.config


def _customize_recording_config(
    base_config: object,
    wrist_wh: tuple[int, int],
    overhead_wh: tuple[int, int],
    *,
    overrides: TeleopConfigOverrides | None = None,
    profile_path: str | None = None,
    env_id: str | None = None,
    factory: ConfigFactory | None = None,
) -> ConfigFactoryUpdate:
    """Apply profile/UI/factory customization and wire recording cameras."""
    config = base_config
    if profile_path is not None and env_id is not None:
        config = apply_config_overrides(
            config,
            load_profile_overrides(profile_path, env_id, config),
        )
    if overrides is not None:
        config = apply_config_overrides(config, overrides)

    factory_update = apply_config_factory(factory, env_id or "", config)
    config = factory_update.config
    if config is None:
        return factory_update

    observations: list | None = getattr(config, "observations", None)
    if observations is None:
        return ConfigFactoryUpdate(config, factory_update.kwargs)

    updated_obs = _wire_camera_observations(observations, wrist_wh, overhead_wh)
    config_attrs = vars(config).copy()
    config_attrs["observations"] = updated_obs
    return ConfigFactoryUpdate(config.__class__(**config_attrs), factory_update.kwargs)


def make_review_video(images: list[np.ndarray], fps: int) -> str | None:
    """Write *images* to a temporary MP4 file and return its path."""
    if not images:
        return None
    from so101_nexus.visualization import save_video

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video:
        path = temp_video.name
    save_video(images, path, fps=fps)
    return path


def make_state_plot(
    states: list[np.ndarray],
    joint_names: tuple[str, ...],
    fps: int,
    rewards: list[float] | None = None,
):
    """Return a Plotly figure showing joint state and per-step reward trajectories."""
    import plotly.graph_objects as go

    arr = np.array(states)
    t = np.arange(arr.shape[0]) / fps
    fig = go.Figure()
    for j, name in enumerate(joint_names):
        fig.add_trace(go.Scatter(x=t, y=arr[:, j], mode="lines", name=name))
    if rewards:
        reward_arr = np.asarray(rewards[: arr.shape[0]], dtype=np.float32)
        fig.add_trace(
            go.Scatter(
                x=t[: reward_arr.shape[0]],
                y=reward_arr,
                mode="lines+markers",
                name="step reward",
                yaxis="y2",
            )
        )
    fig.update_layout(
        title="Joint States and Step Reward Over Time",
        xaxis_title="Time (s)",
        yaxis_title="Position (deg / RANGE_0_100)",
        yaxis2={
            "title": "Step reward",
            "overlaying": "y",
            "side": "right",
        },
        height=440,
    )
    return fig
