"""Dataset field selection, feature schema, and per-frame builders.

The selection surfaces in the Gradio UI as checkboxes. ``observation.state``
and ``action`` are always written; they are not user-toggleable. Camera fields
can be opted out of, in which case they are excluded from both the declared
LeRobot feature schema and every recorded frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

import numpy as np

from so101_nexus.config import REWARD_COMPONENT_KEYS

WRIST_KEY = "observation.images.wrist"
OVERHEAD_KEY = "observation.images.overhead"
SIDE_KEY = "observation.images.side"
_CAMERA_FIELDS = (
    ("wrist", WRIST_KEY, "check that the env exposes a wrist camera."),
    ("overhead", OVERHEAD_KEY, "check that the env exposes an overhead camera."),
    ("side", SIDE_KEY, "check that the env renders with render_mode='rgb_array'."),
)
ENV_STATE_KEY = "observation.environment_state"
REWARD_KEY = "reward"
SUCCESS_KEY = "success"
DONE_KEY = "done"
REWARD_COMPONENTS_KEY_PREFIX = "reward_components."
# Canonical LeRobot scalar feature (matches the `modify_features` docstring
# example in lerobot.datasets.dataset_tools): a scalar float32 per frame. Reused
# for reward, success, and done, which are all always-recorded env-step scalars.
SCALAR_FEATURE: dict[str, Any] = {"dtype": "float32", "shape": (1,), "names": None}
REWARD_FEATURE: dict[str, Any] = SCALAR_FEATURE
# Per-facet reward breakdown (see `RewardConfig.compute_components` /
# `compute_simple_components`), one always-recorded scalar per
# `REWARD_COMPONENT_KEYS` entry so callers can inspect e.g. grasping or
# penalty contributions separately from the summed `reward` key.
REWARD_COMPONENT_FEATURE_KEYS: tuple[str, ...] = tuple(
    f"{REWARD_COMPONENTS_KEY_PREFIX}{name}" for name in REWARD_COMPONENT_KEYS
)
# The always-recorded per-step env scalars, in schema order.
SCALAR_KEYS: tuple[str, ...] = (REWARD_KEY, SUCCESS_KEY, DONE_KEY, *REWARD_COMPONENT_FEATURE_KEYS)


def save_recorded_episode(
    dataset: Any,
    placement_contract: dict | None = None,
    *,
    reproducibility: dict[str, Any] | None = None,
) -> None:
    """Save an episode with evaluator and reproducibility metadata."""
    import json
    from pathlib import Path

    metadata_items = [
        (directory_name, json.dumps(metadata, indent=2, allow_nan=False))
        for directory_name, metadata in (
            ("placement_contracts", placement_contract),
            ("reproducibility", reproducibility),
        )
        if metadata is not None
    ]
    if not metadata_items:
        dataset.save_episode()
        return

    root = Path(dataset.root)
    paths: list[Path] = []
    try:
        for directory_name, serialized in metadata_items:
            directory = root / "meta" / directory_name
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"episode_{dataset.num_episodes:06d}.json"
            with path.open("x") as stream:
                paths.append(path)
                stream.write(serialized)
        dataset.save_episode()
    except Exception:
        for path in paths:
            path.unlink()
        raise


@dataclass(frozen=True)
class FieldSelection:
    """Which fields the user opted to persist into the LeRobot dataset."""

    wrist_image: bool = True
    overhead_image: bool = True
    side_image: bool = False
    environment_state: bool = True
    task: bool = True

    @property
    def state(self) -> bool:
        """``observation.state`` is always persisted."""
        return True

    @property
    def action(self) -> bool:
        """``action`` is always persisted."""
        return True


def _with_selected_cameras(
    selection: FieldSelection,
    follower_features: dict[str, Any],
) -> dict[str, Any]:
    """Return follower observation features with deselected cameras removed."""
    features = {key: value for key, value in follower_features.items() if value is float}
    for name, _, _ in _CAMERA_FIELDS:
        if getattr(selection, f"{name}_image"):
            if name not in follower_features:
                raise ValueError(f"{name} image selected but follower exposes no '{name}' camera.")
            features[name] = follower_features[name]
    return features


def build_features(
    selection: FieldSelection,
    follower_features: dict[str, Any],
    action_features: dict[str, Any],
    *,
    env_state_names: list[str] | None = None,
    intervention: bool = False,
) -> dict[str, dict[str, Any]]:
    """Return the LeRobot feature dict for the selected fields.

    Parameters
    ----------
    selection
        Which fields the user chose to persist.
    follower_features
        ``SimSOFollower.observation_features``-shaped dict.
    action_features
        ``SimSOFollower.action_features``-shaped dict.
    env_state_names
        Per-dimension names for the privileged ``observation.environment_state``
        vector (see :func:`so101_nexus.observations.privileged_state_feature_names`).
        When provided and ``selection.environment_state`` is set, the channel is
        declared as a ``(len(names),)`` float32 feature; otherwise it is omitted.
    intervention
        Include human-control flags and intervention segment IDs for policy collection.
    """
    from lerobot.utils.feature_utils import hw_to_dataset_features

    features: dict[str, dict[str, Any]] = {}
    features.update(hw_to_dataset_features(action_features, "action", use_video=True))
    features.update(
        hw_to_dataset_features(
            _with_selected_cameras(selection, follower_features),
            "observation",
            use_video=True,
        )
    )
    # Privileged low-dim ground-truth state (TCP/object pose, grasp, offsets).
    # Declared directly because `hw_to_dataset_features` would fold per-dim
    # floats into `observation.state`; a standalone vector needs its own key.
    if selection.environment_state and env_state_names:
        features[ENV_STATE_KEY] = {
            "dtype": "float32",
            "shape": (len(env_state_names),),
            "names": list(env_state_names),
        }
    # reward/success/done are always available from `env.step` and are always
    # recorded; they are not user-toggleable fields, so declared unconditionally.
    for key in SCALAR_KEYS:
        features[key] = dict(SCALAR_FEATURE)
    if intervention:
        features["is_intervention"] = dict(SCALAR_FEATURE)
        features["intervention_id"] = {"dtype": "int64", "shape": (1,), "names": None}
    return features


def build_frame(
    selection: FieldSelection,
    *,
    state: np.ndarray,
    action: np.ndarray,
    task: str,
    reward: float = 0.0,
    reward_components: Mapping[str, float] | None = None,
    success: float = 0.0,
    done: float = 0.0,
    env_state: np.ndarray | None = None,
    wrist_image: np.ndarray | None,
    overhead_image: np.ndarray | None,
    side_image: np.ndarray | None = None,
    intervention_id: int | None = None,
) -> dict[str, Any]:
    """Assemble one LeRobot frame dict using only the selected fields."""
    components = reward_components or {}
    frame: dict[str, Any] = {
        "observation.state": state.astype(np.float32),
        "action": action.astype(np.float32),
        # Per-step env-step scalars, aligned with `action` on this frame
        # (LeRobot `Transition` carries reward with (observation, action)).
        REWARD_KEY: np.array([reward], dtype=np.float32),
        SUCCESS_KEY: np.array([success], dtype=np.float32),
        DONE_KEY: np.array([done], dtype=np.float32),
        # Per-facet breakdown of `reward` (see `RewardConfig.compute_components`);
        # missing keys (e.g. a caller with no component data) default to 0.0 so
        # every frame still satisfies LeRobot's exact declared-feature match.
        **{
            key: np.array([components.get(name, 0.0)], dtype=np.float32)
            for name, key in zip(REWARD_COMPONENT_KEYS, REWARD_COMPONENT_FEATURE_KEYS, strict=True)
        },
    }
    # LeRobot v3 stores task text outside the regular feature schema, but
    # `LeRobotDataset.add_frame` requires it on every frame.
    frame["task"] = task
    images = {"wrist": wrist_image, "overhead": overhead_image, "side": side_image}
    for name, key, hint in _CAMERA_FIELDS:
        if getattr(selection, f"{name}_image"):
            image = images[name]
            if image is None:
                raise ValueError(f"{name}_image selected but no {name} frame was recorded; {hint}")
            frame[key] = image
    if selection.environment_state:
        if env_state is None:
            raise ValueError(
                "environment_state selected but no privileged state was recorded; "
                "check that the env exposes non-camera observation components."
            )
        frame[ENV_STATE_KEY] = np.asarray(env_state, dtype=np.float32)
    if intervention_id is not None:
        if intervention_id < 0:
            raise ValueError("intervention_id must be nonnegative")
        frame["is_intervention"] = np.array([intervention_id > 0], dtype=np.float32)
        frame["intervention_id"] = np.array([intervention_id], dtype=np.int64)
    return frame


def _make_reward_scalar_dataset_cls():
    """Return a LeRobotDataset subclass that stores reward as HF scalars.

    LeRobot validates (1,)-shaped features against ndarrays of shape (1,) in
    ``validate_frame``, then maps them to HuggingFace scalar
    ``Value("float32")`` in ``get_hf_features_from_features``. The buffer of
    (1,) arrays that ``add_frame`` accumulates triggers a NumPy deprecation
    warning when ``datasets`` coerces each element to ``float`` during
    ``save_episode``. This subclass squeezes each scalar buffer (reward,
    success, done) to Python scalars before the ``np.stack`` that backs
    ``save_episode``, matching how LeRobot handles its own ``DEFAULT_FEATURES``
    (timestamp, frame_index, etc.).

    LeRobot 0.6 records through ``DatasetWriter.episode_buffer``. Only
    ``save_episode`` needs overriding; other methods already use that buffer.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    class RewardRecordingDataset(LeRobotDataset):
        """LeRobotDataset that records scalar features without coercion warnings."""

        def save_episode(
            self,
            episode_data: dict | None = None,
            parallel_encoding: bool = True,
        ) -> None:
            if episode_data is None:
                buf = getattr(getattr(self, "writer", None), "episode_buffer", None)
                if buf is not None:
                    for key in (*SCALAR_KEYS, "is_intervention", "intervention_id"):
                        if isinstance(buf.get(key), list):
                            buf[key] = [np.asarray(v).reshape(-1)[0].item() for v in buf[key]]
            super().save_episode(episode_data, parallel_encoding)

    return RewardRecordingDataset
