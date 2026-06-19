"""Teleop environment customization helpers.

These helpers keep teleop customization in pure Python data structures so
the Gradio UI, CLI flags, and profile files all end up producing the same
typed config objects that regular SO101-Nexus examples pass to ``gym.make``.
"""

from __future__ import annotations

import importlib
import json
import math
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, cast

from so101_nexus.constants import (
    COLOR_MAP,
    CUBE_COLOR_MAP,
    YCB_OBJECTS,
    ColorConfig,
    ColorName,
)
from so101_nexus.objects import CubeObject, MeshObject, SceneObject, YCBObject

ConfigFactory = Callable[[str, object | None], object | dict[str, Any]]


@dataclass(frozen=True)
class TeleopConfigOverrides:
    """Optional environment config overrides collected from UI, CLI, or profiles."""

    objects: tuple[SceneObject, ...] | None = None
    object_specs: tuple[str, ...] | None = None
    n_distractors: int | None = None
    ground_colors: tuple[ColorName, ...] | None = None
    robot_colors: tuple[ColorName, ...] | None = None
    spawn_min_radius: float | None = None
    spawn_max_radius: float | None = None
    spawn_angle_half_range_deg: float | None = None
    reset_settle_frames: int | None = None
    cube_colors: tuple[ColorName, ...] | None = None
    target_colors: tuple[ColorName, ...] | None = None

    def __post_init__(self) -> None:
        """Validate override combinations that dataclass types cannot express."""
        if self.n_distractors is not None and self.n_distractors < 0:
            raise ValueError(f"n_distractors must be >= 0, got {self.n_distractors}")
        if self.reset_settle_frames is not None and self.reset_settle_frames < 0:
            raise ValueError(f"reset_settle_frames must be >= 0, got {self.reset_settle_frames}")
        if self.objects is not None and self.object_specs is not None:
            raise ValueError("Provide either objects or object_specs, not both.")


@dataclass(frozen=True)
class ConfigFactoryUpdate:
    """Config and gym kwargs returned by a custom teleop config factory."""

    config: object | None
    kwargs: dict[str, Any]


_OVERRIDE_KEYS = {field.name for field in fields(TeleopConfigOverrides)}
_PROFILE_SECTION_KEYS = {"common", "pick", "pick_and_place", "envs"}


def default_color_choices() -> list[str]:
    """Return valid color names for teleop UI controls."""
    return list(COLOR_MAP)


def default_cube_color_choices() -> list[str]:
    """Return color names suitable for movable cubes and target markers."""
    return list(CUBE_COLOR_MAP)


def default_object_choices() -> list[str]:
    """Return built-in object spec strings for teleop UI controls."""
    return [f"cube:{color}" for color in COLOR_MAP] + [
        f"ycb:{model_id}" for model_id in YCB_OBJECTS
    ]


def color_tuple_from_names(
    values: Sequence[str],
    *,
    field_name: str,
    valid_colors: Mapping[str, object] = COLOR_MAP,
) -> tuple[ColorName, ...] | None:
    """Validate UI-selected color names and return them as typed literals."""
    if not values:
        return None
    return tuple(
        _validate_color(str(value), field_name=field_name, valid_colors=valid_colors)
        for value in values
    )


def object_from_spec(spec: str) -> SceneObject:
    """Build a built-in scene object from a compact UI/CLI spec string."""
    kind, *parts = spec.split(":")
    if kind == "cube" and len(parts) == 1:
        return CubeObject(color=_validate_color(parts[0], field_name="cube color"))
    if kind == "ycb" and len(parts) == 1:
        return YCBObject(model_id=parts[0])
    if kind == "mesh":
        raise ValueError("mesh objects must use mapping syntax to avoid ambiguous path parsing")
    raise ValueError(f"object spec {spec!r} must be cube:<color> or ycb:<model_id>")


def object_from_mapping(raw: Mapping[str, Any]) -> SceneObject:
    """Build a scene object from a profile mapping."""
    kind = raw.get("type")
    if kind == "cube":
        return CubeObject(
            color=_validate_color(str(_required(raw, "color")), field_name="cube color")
        )
    if kind == "ycb":
        return YCBObject(model_id=str(_required(raw, "model_id")))
    if kind == "mesh":
        return MeshObject(
            collision_mesh_path=str(_required(raw, "collision_mesh_path")),
            visual_mesh_path=str(_required(raw, "visual_mesh_path")),
            mass=_as_float(_required(raw, "mass"), "mass"),
            name=str(_required(raw, "name")),
            scale=_as_float(raw.get("scale", 1.0), "scale"),
        )
    raise ValueError(f"unsupported object mapping type: {kind!r}")


def apply_config_overrides[ConfigT](
    config: ConfigT,
    overrides: TeleopConfigOverrides,
) -> ConfigT:
    """Return a cloned config with applicable teleop overrides applied."""
    attrs = vars(config).copy()
    is_pick_like = "objects" in attrs and "n_distractors" in attrs

    for name in ("ground_colors", "robot_colors"):
        value = _as_color_config(getattr(overrides, name))
        if value is not None and name in attrs:
            attrs[name] = value

    for name in (
        "spawn_min_radius",
        "spawn_max_radius",
        "spawn_angle_half_range_deg",
        "reset_settle_frames",
    ):
        value = getattr(overrides, name)
        if value is not None and name in attrs:
            attrs[name] = value

    if is_pick_like and overrides.objects is not None:
        attrs["objects"] = list(overrides.objects)
    elif is_pick_like and overrides.object_specs is not None:
        attrs["objects"] = [object_from_spec(spec) for spec in overrides.object_specs]

    if overrides.n_distractors is not None and "n_distractors" in attrs:
        attrs["n_distractors"] = overrides.n_distractors

    for name in ("cube_colors", "target_colors"):
        value = _as_color_config(getattr(overrides, name))
        if value is not None and name in attrs:
            attrs[name] = value

    return config.__class__(**attrs)


def load_profile_overrides(
    path: str | Path,
    env_id: str,
    base_config: object,
) -> TeleopConfigOverrides:
    """Load config overrides for *env_id* from a JSON or TOML profile."""
    profile = _load_mapping(Path(path))
    _validate_profile_keys(profile)
    merged: dict[str, Any] = {key: profile[key] for key in set(profile) & _OVERRIDE_KEYS}
    merged.update(_mapping_section(profile, "common"))

    base_attrs = vars(base_config)
    if "cube_colors" in base_attrs:
        merged.update(_mapping_section(profile, "pick_and_place"))
    elif "objects" in base_attrs and "n_distractors" in base_attrs:
        merged.update(_mapping_section(profile, "pick"))

    envs = profile.get("envs", {})
    if envs is not None and not isinstance(envs, Mapping):
        raise ValueError("profile key 'envs' must be a mapping")
    if isinstance(envs, Mapping):
        env_section = envs.get(env_id, {})
        if not isinstance(env_section, Mapping):
            raise ValueError(f"profile envs.{env_id!r} must be a mapping")
        merged.update(env_section)

    return overrides_from_mapping(merged)


def overrides_from_mapping(raw: Mapping[str, Any]) -> TeleopConfigOverrides:
    """Convert a profile mapping into validated teleop overrides."""
    _validate_override_keys(raw)
    kwargs: dict[str, Any] = {}
    if "objects" in raw:
        kwargs["objects"] = tuple(_objects_from_profile(raw["objects"]))
    if "object_specs" in raw:
        kwargs["object_specs"] = tuple(str(spec) for spec in _as_sequence(raw["object_specs"]))
    if "n_distractors" in raw:
        kwargs["n_distractors"] = _as_nonnegative_int(raw["n_distractors"], "n_distractors")
    for key in ("ground_colors", "robot_colors", "cube_colors", "target_colors"):
        if key in raw:
            kwargs[key] = _color_tuple(raw[key], key)
    for key in ("spawn_min_radius", "spawn_max_radius", "spawn_angle_half_range_deg"):
        if key in raw:
            kwargs[key] = _as_float(raw[key], key)
    if "reset_settle_frames" in raw:
        kwargs["reset_settle_frames"] = _as_nonnegative_int(
            raw["reset_settle_frames"], "reset_settle_frames"
        )
    return TeleopConfigOverrides(**kwargs)


def object_to_mapping(obj: SceneObject) -> dict[str, Any]:
    """Serialize a scene object to a profile mapping (inverse of :func:`object_from_mapping`)."""
    if isinstance(obj, CubeObject):
        return {
            "type": "cube",
            "color": obj.color,
        }  # ponytail: schema drops mass, see object_from_mapping
    if isinstance(obj, YCBObject):
        return {"type": "ycb", "model_id": obj.model_id}
    if isinstance(obj, MeshObject):
        return {
            "type": "mesh",
            "collision_mesh_path": obj.collision_mesh_path,
            "visual_mesh_path": obj.visual_mesh_path,
            "mass": obj.mass,
            "name": obj.name,
            "scale": obj.scale,
        }
    raise ValueError(f"cannot serialize unsupported scene object: {obj!r}")


def overrides_to_mapping(overrides: TeleopConfigOverrides) -> dict[str, Any]:
    """Serialize overrides to a profile mapping (inverse of :func:`overrides_from_mapping`)."""
    mapping: dict[str, Any] = {}
    if overrides.objects is not None:
        mapping["objects"] = [object_to_mapping(obj) for obj in overrides.objects]
    if overrides.object_specs is not None:
        mapping["object_specs"] = list(overrides.object_specs)
    for key in (
        "n_distractors",
        "ground_colors",
        "robot_colors",
        "cube_colors",
        "target_colors",
        "spawn_min_radius",
        "spawn_max_radius",
        "spawn_angle_half_range_deg",
        "reset_settle_frames",
    ):
        value = getattr(overrides, key)
        if value is not None:
            mapping[key] = list(value) if isinstance(value, tuple) else value
    return mapping


def load_config_factory(ref: str | None) -> ConfigFactory | None:
    """Resolve a ``module:function`` config factory reference."""
    if not ref:
        return None
    module_name, sep, attr_name = ref.partition(":")
    if not sep:
        raise ValueError("config factory must use 'module:function'")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ValueError(f"could not import --env-config-factory module {module_name!r}") from exc
    try:
        func = getattr(module, attr_name)
    except AttributeError as exc:
        raise ValueError(f"--env-config-factory {ref!r} does not exist") from exc
    if not callable(func):
        raise TypeError(f"config factory {ref!r} is not callable")
    return cast("ConfigFactory", func)


def apply_config_factory(
    factory: ConfigFactory | None,
    env_id: str,
    base_config: object | None,
) -> ConfigFactoryUpdate:
    """Evaluate a custom config factory without mutating caller-owned kwargs."""
    if factory is None:
        return ConfigFactoryUpdate(base_config, {})
    result = factory(env_id, base_config)
    if isinstance(result, dict):
        updates = dict(cast("dict[str, Any]", result))
        return ConfigFactoryUpdate(updates.pop("config", base_config), updates)
    if result is None:
        raise ValueError("config factory must return a config object or a dict of gym kwargs")
    return ConfigFactoryUpdate(result, {})


def _load_mapping(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    suffix = path.suffix.lower()
    if suffix == ".json":
        loaded = json.loads(data.decode("utf-8"))
    elif suffix == ".toml":
        loaded = tomllib.loads(data.decode("utf-8"))
    else:
        raise ValueError("teleop config profile must be .json or .toml")
    if not isinstance(loaded, dict):
        raise ValueError("teleop config profile must contain a mapping at the top level")
    return loaded


def _validate_profile_keys(profile: Mapping[str, Any]) -> None:
    unknown = set(profile) - _OVERRIDE_KEYS - _PROFILE_SECTION_KEYS
    if unknown:
        keys = ", ".join(sorted(unknown))
        raise ValueError(f"unknown teleop config profile key(s): {keys}")


def _validate_override_keys(raw: Mapping[str, Any]) -> None:
    unknown = set(raw) - _OVERRIDE_KEYS
    if unknown:
        keys = ", ".join(sorted(unknown))
        raise ValueError(f"unknown teleop config override key(s): {keys}")


def _mapping_section(profile: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = profile.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"profile key {key!r} must be a mapping")
    return value


def _objects_from_profile(value: object) -> list[SceneObject]:
    objects: list[SceneObject] = []
    for item in _as_sequence(value):
        if isinstance(item, Mapping):
            objects.append(object_from_mapping(cast("Mapping[str, Any]", item)))
        elif isinstance(item, str):
            objects.append(object_from_spec(item))
        else:
            raise ValueError(f"profile objects entries must be mappings or strings, got {item!r}")
    return objects


def _as_sequence(value: object) -> Sequence[object]:
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, Sequence):
        raise ValueError(f"expected a sequence, got {value!r}")
    return value


def _as_nonnegative_int(value: object, key: str) -> int:
    if not isinstance(value, int):
        raise ValueError(f"{key} must be an integer, got {value!r}")
    if value < 0:
        raise ValueError(f"{key} must be >= 0, got {value}")
    return value


def _as_float(value: object, key: str) -> float:
    if not isinstance(value, str | int | float):
        raise ValueError(f"{key} must be numeric, got {value!r}")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be numeric, got {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{key} must be finite, got {value!r}")
    return parsed


def _color_tuple(value: object, key: str) -> tuple[ColorName, ...]:
    return tuple(_validate_color(str(item), field_name=key) for item in _as_sequence(value))


def _as_color_config(value: tuple[ColorName, ...] | None) -> ColorConfig | None:
    if value is None:
        return None
    return list(value)


def _validate_color(
    value: str,
    *,
    field_name: str = "color",
    valid_colors: Mapping[str, object] = COLOR_MAP,
) -> ColorName:
    if value not in valid_colors:
        raise ValueError(f"unknown {field_name} {value!r}; expected one of {list(valid_colors)}")
    return cast("ColorName", value)


def _required(raw: Mapping[str, Any], key: str) -> object:
    if key not in raw:
        raise ValueError(f"object mapping missing required key {key!r}")
    return raw[key]
