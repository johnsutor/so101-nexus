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
import inspect
import tempfile

import numpy as np

from so101_nexus_core.observations import OverheadCamera, WristCamera


def _default_repo_id(env_id: str) -> str:
    """Generate a local-only dataset repo ID from *env_id* and the current timestamp."""
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = env_id.replace("/", "-").replace(" ", "_")
    return f"local/teleop-{safe}-{ts}"


def _resolve_env_config(env_ctor: type) -> object | None:
    """Resolve the default config object for an environment class, or ``None``."""
    config_param = inspect.signature(env_ctor.__init__).parameters.get("config")
    if config_param is None:
        return None

    base_config = config_param.default
    if base_config is not None and base_config is not inspect.Parameter.empty:
        return base_config

    config_class_name: object = None
    for cls in inspect.getmro(env_ctor):
        if hasattr(cls, "__annotations__") and "config" in cls.__annotations__:
            config_class_name = cls.__annotations__["config"]
            break

    if not isinstance(config_class_name, str):
        return None

    try:
        from so101_nexus_core import config as config_module

        config_class = getattr(config_module, config_class_name, None)
        if config_class is not None:
            return config_class()
    except Exception:
        return None
    return None


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

    from so101_nexus_core.teleop.leader import import_backend_for_env_id

    import_backend_for_env_id(env_id)
    spec = gym.spec(env_id)
    entry_point = spec.entry_point
    if isinstance(entry_point, str):
        module_name, attr_name = entry_point.split(":")
        env_ctor = getattr(importlib.import_module(module_name), attr_name)
    else:
        env_ctor = entry_point
    return env_ctor, dict(spec.kwargs)


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
) -> dict:
    """Return ``gym.make`` kwargs for teleop recording with both cameras sized."""
    env_ctor, kwargs = _resolve_env_ctor(env_id)
    if not inspect.isclass(env_ctor):
        return kwargs

    base_config = _resolve_env_config(env_ctor)
    if base_config is None:
        return kwargs

    observations: list | None = getattr(base_config, "observations", None)
    if observations is None:
        return kwargs

    updated_obs = _wire_camera_observations(observations, wrist_wh, overhead_wh)
    config_attrs = vars(base_config).copy()
    config_attrs["observations"] = updated_obs
    updated_config = base_config.__class__(**config_attrs)
    kwargs["config"] = updated_config
    return kwargs


def make_review_video(images: list[np.ndarray], fps: int) -> str | None:
    """Write *images* to a temporary MP4 file and return its path."""
    if not images:
        return None
    from so101_nexus_core.visualization import save_video

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video:
        path = temp_video.name
    save_video(images, path, fps=fps)
    return path


def make_state_plot(states: list[np.ndarray], joint_names: tuple[str, ...], fps: int):
    """Return a Plotly figure showing joint state trajectories over time."""
    import plotly.graph_objects as go

    arr = np.array(states)
    t = np.arange(arr.shape[0]) / fps
    fig = go.Figure()
    for j, name in enumerate(joint_names):
        fig.add_trace(go.Scatter(x=t, y=arr[:, j], mode="lines", name=name))
    fig.update_layout(
        title="Joint States Over Time",
        xaxis_title="Time (s)",
        yaxis_title="Position (rad)",
        height=400,
    )
    return fig
