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
ENV_STATE_KEY = "observation.environment_state"
REWARD_KEY = "reward"
SUCCESS_KEY = "success"
DONE_KEY = "done"
# Canonical LeRobot scalar feature (matches the `modify_features` docstring
# example in lerobot.datasets.dataset_tools): a scalar float32 per frame. Reused
# for reward, success, and done, which are all always-recorded env-step scalars.
SCALAR_FEATURE: dict[str, Any] = {"dtype": "float32", "shape": (1,), "names": None}
REWARD_FEATURE: dict[str, Any] = SCALAR_FEATURE
# The always-recorded per-step env scalars, in schema order.
SCALAR_KEYS: tuple[str, ...] = (REWARD_KEY, SUCCESS_KEY, DONE_KEY)


@dataclass(frozen=True)
class FieldSelection:
    """Which fields the user opted to persist into the LeRobot dataset."""

    wrist_image: bool = True
    overhead_image: bool = True
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
    *,
    env_state_names: list[str] | None = None,
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
    return features


def build_frame(
    selection: FieldSelection,
    *,
    state: np.ndarray,
    action: np.ndarray,
    task: str,
    reward: float = 0.0,
    success: float = 0.0,
    done: float = 0.0,
    env_state: np.ndarray | None = None,
    wrist_image: np.ndarray | None,
    overhead_image: np.ndarray | None,
) -> dict[str, Any]:
    """Assemble one LeRobot frame dict using only the selected fields."""
    frame: dict[str, Any] = {
        "observation.state": state.astype(np.float32),
        "action": action.astype(np.float32),
        # Per-step env-step scalars, aligned with `action` on this frame
        # (LeRobot `Transition` carries reward with (observation, action)).
        REWARD_KEY: np.array([reward], dtype=np.float32),
        SUCCESS_KEY: np.array([success], dtype=np.float32),
        DONE_KEY: np.array([done], dtype=np.float32),
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
    if selection.environment_state:
        if env_state is None:
            raise ValueError(
                "environment_state selected but no privileged state was recorded; "
                "check that the env exposes non-camera observation components."
            )
        frame[ENV_STATE_KEY] = np.asarray(env_state, dtype=np.float32)
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

    The in-progress buffer moved between supported 0.5.x layouts: 0.5.0 keeps
    it on the dataset (``self.episode_buffer``), while 0.5.1 routes recording
    through a ``DatasetWriter`` reached via ``self.writer.episode_buffer``.
    Only ``save_episode`` needs overriding: ``add_frame`` and
    ``clear_episode_buffer`` already act on the live buffer on both layouts
    (native in 0.5.0, writer-delegated in 0.5.1) and never stack reward.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    class RewardRecordingDataset(LeRobotDataset):
        """LeRobotDataset that records scalar features without coercion warnings."""

        def _active_episode_buffer(self) -> dict | None:
            """Return the in-progress episode buffer across LeRobot 0.5.x layouts."""
            buf = getattr(self, "episode_buffer", None)
            if buf is None:
                buf = getattr(getattr(self, "writer", None), "episode_buffer", None)
            return buf

        def save_episode(
            self,
            episode_data: dict | None = None,
            parallel_encoding: bool = True,
        ) -> None:
            if episode_data is None:
                buf = self._active_episode_buffer()
                if buf is not None:
                    for key in SCALAR_KEYS:
                        if isinstance(buf.get(key), list):
                            buf[key] = [float(np.asarray(v).reshape(-1)[0]) for v in buf[key]]
            super().save_episode(episode_data, parallel_encoding)

    return RewardRecordingDataset
