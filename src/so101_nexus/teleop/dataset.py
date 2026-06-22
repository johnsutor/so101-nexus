"""Dataset field selection, feature schema, and per-frame builders.

The selection surfaces in the Gradio UI as checkboxes. ``observation.state``
and ``action`` are always written; they are not user-toggleable. Camera fields
can be opted out of, in which case they are excluded from both the declared
LeRobot feature schema and every recorded frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any

import numpy as np

WRIST_KEY = "observation.images.wrist"
OVERHEAD_KEY = "observation.images.overhead"
REWARD_KEY = "reward"
# Canonical LeRobot reward feature (matches the `modify_features` docstring
# example in lerobot.datasets.dataset_tools): a scalar float32 per frame.
REWARD_FEATURE: dict[str, Any] = {"dtype": "float32", "shape": (1,), "names": None}


@dataclass(frozen=True)
class FieldSelection:
    """Which fields the user opted to persist into the LeRobot dataset."""

    wrist_image: bool = True
    overhead_image: bool = True
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
    if selection.wrist_image:
        if "wrist" not in follower_features:
            raise ValueError("wrist image selected but follower exposes no 'wrist' camera.")
        features["wrist"] = follower_features["wrist"]
    if selection.overhead_image:
        if "overhead" not in follower_features:
            raise ValueError("overhead image selected but follower exposes no 'overhead' camera.")
        features["overhead"] = follower_features["overhead"]
    return features


def _hw_to_dataset_features():
    """Return LeRobot's feature-schema converter across supported 0.5.x layouts."""
    try:
        return import_module("lerobot.datasets.feature_utils").hw_to_dataset_features
    except (ImportError, AttributeError):  # LeRobot 0.5.0 compatibility
        return import_module("lerobot.datasets.utils").hw_to_dataset_features


def build_features(
    selection: FieldSelection,
    follower_features: dict[str, Any],
    action_features: dict[str, Any],
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
    """
    hw_to_dataset_features = _hw_to_dataset_features()
    features: dict[str, dict[str, Any]] = {}
    features.update(hw_to_dataset_features(action_features, "action", use_video=True))
    features.update(
        hw_to_dataset_features(
            _with_selected_cameras(selection, follower_features),
            "observation",
            use_video=True,
        )
    )
    # Reward is always available from `env.step` and is always recorded; it is
    # not a user-toggleable field, so it is declared unconditionally.
    features[REWARD_KEY] = dict(REWARD_FEATURE)
    return features


def build_frame(
    selection: FieldSelection,
    *,
    state: np.ndarray,
    action: np.ndarray,
    task: str,
    reward: float = 0.0,
    wrist_image: np.ndarray | None,
    overhead_image: np.ndarray | None,
) -> dict[str, Any]:
    """Assemble one LeRobot frame dict using only the selected fields."""
    frame: dict[str, Any] = {
        "observation.state": state.astype(np.float32),
        "action": action.astype(np.float32),
        # Per-step transition reward, aligned with `action` on this frame
        # (LeRobot `Transition` carries reward with (observation, action)).
        REWARD_KEY: np.array([reward], dtype=np.float32),
    }
    # LeRobot v3 stores task text outside the regular feature schema, but
    # `LeRobotDataset.add_frame` requires it on every frame.
    frame["task"] = task
    if selection.wrist_image:
        if wrist_image is None:
            raise ValueError(
                "wrist_image selected but no wrist frame was recorded; "
                "check that the env exposes a wrist camera."
            )
        frame[WRIST_KEY] = wrist_image
    if selection.overhead_image:
        if overhead_image is None:
            raise ValueError(
                "overhead_image selected but no overhead frame was recorded; "
                "check that the env exposes an overhead camera."
            )
        frame[OVERHEAD_KEY] = overhead_image
    return frame


def _make_reward_scalar_dataset_cls():
    """Return a LeRobotDataset subclass that stores reward as HF scalars.

    LeRobot validates (1,)-shaped features against ndarrays of shape (1,) in
    ``validate_frame``, then maps them to HuggingFace scalar
    ``Value("float32")`` in ``get_hf_features_from_features``. The buffer of
    (1,) arrays that ``add_frame`` accumulates triggers a NumPy deprecation
    warning when ``datasets`` coerces each element to ``float`` during
    ``save_episode``. This subclass squeezes the reward buffer to Python
    scalars before the parent's ``np.stack``, matching how LeRobot handles
    its own ``DEFAULT_FEATURES`` (timestamp, frame_index, etc.).
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    class RewardRecordingDataset(LeRobotDataset):
        """LeRobotDataset that records the reward feature without scalar coercion warnings."""

        def save_episode(
            self,
            episode_data: dict | None = None,
            parallel_encoding: bool = True,
        ) -> None:
            if episode_data is None and self.episode_buffer is not None:
                buf = self.episode_buffer
                if "reward" in buf and isinstance(buf["reward"], list):
                    buf["reward"] = [float(np.asarray(v).reshape(-1)[0]) for v in buf["reward"]]
            super().save_episode(episode_data, parallel_encoding)

    return RewardRecordingDataset
