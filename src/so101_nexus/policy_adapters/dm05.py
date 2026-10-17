"""SO101 action collection through OpenDM's isolated inference runtime."""

from __future__ import annotations

import base64
import json
from collections import deque
from io import BytesIO
from typing import Any
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import numpy as np


class DM05PolicyAdapter:
    """Replay absolute SO101 targets from an OpenDM SO101 inference server.

    Parameters
    ----------
    url
        Full HTTP endpoint, including ``/v1/infer``. The server owns checkpoint
        loading, normalization, and relative-to-absolute action conversion.
    fps
        Action rate from the training dataset, also selected in the recorder.
    camera_map
        Collector image keys mapped to ``observation.images.images_1`` (head)
        and ``observation.images.images_2`` (wrist).
    """

    def __init__(
        self,
        url: str,
        *,
        fps: int | None = None,
        camera_map: dict[str, str] | None = None,
    ) -> None:
        parsed = urlsplit(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError(
                "DM0.5 requires an OpenDM /v1/infer URL as --policy-path; "
                "start its SO101 server with Dexmal/DM05-SO101-Pick-Cube weights"
            )
        if fps is None or fps <= 0:
            raise ValueError("DM0.5 requires --policy-fps from its training dataset")
        self.url = url
        self.fps = fps
        mapping = camera_map or {
            "observation.images.overhead": "observation.images.images_1",
            "observation.images.wrist": "observation.images.images_2",
        }
        targets = ["observation.images.images_1", "observation.images.images_2"]
        if len(mapping) != 2 or set(mapping.values()) != set(targets):
            raise ValueError("DM0.5 camera mappings must select images_1 and images_2 exactly once")
        self.image_keys = [
            next(source for source in mapping if mapping[source] == t) for t in targets
        ]
        self._actions: deque[np.ndarray] = deque()

    def select_action(self, batch: dict[str, Any]) -> np.ndarray:
        """Return one absolute action in degrees, with the gripper in percent."""
        if not self._actions:
            state = np.asarray(batch["observation.state"], dtype=np.float32)
            if state.shape != (6,) or not np.isfinite(state).all():
                raise ValueError("DM0.5 requires a finite six-joint SO101 state")
            missing = set(self.image_keys) - batch.keys()
            if missing:
                raise ValueError(f"Missing DM0.5 cameras: {sorted(missing)}")
            images = {
                str(index): _encode_image(batch[key])
                for index, key in enumerate(self.image_keys, start=1)
            }
            payload = {
                "observation": {
                    "prompt": batch["task"],
                    "robot_type": "SO101",
                    "state": state.tolist(),
                    "images": images,
                }
            }
            request = Request(
                self.url,
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request, timeout=120) as response:
                actions = np.asarray(json.load(response).get("actions"), dtype=np.float32)
            if (
                actions.ndim != 2
                or actions.shape[0] == 0
                or actions.shape[1] != 6
                or not np.isfinite(actions).all()
            ):
                raise ValueError("DM0.5 must return a finite nonempty chunk of six-joint targets")
            self._actions.extend(actions)
        return self._actions.popleft().copy()

    def reset(self) -> None:
        """Discard unexecuted targets at episode and handoff boundaries."""
        self._actions.clear()


def _encode_image(image: Any) -> str:
    from PIL import Image

    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
        raise ValueError("DM0.5 requires HWC uint8 RGB camera images")
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")
