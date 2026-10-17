"""Native LeRobot and remote OpenDM policy collection for DM0.5 SO101."""

from __future__ import annotations

import base64
import json
from collections import deque
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import numpy as np

if TYPE_CHECKING:
    from so101_nexus.policy_adapters.collection import LeRobotPolicyAdapter

_DM05_IMAGE_KEYS = ("observation.images.images_1", "observation.images.images_2")


def _dm05_camera_map(camera_map: dict[str, str] | None) -> dict[str, str]:
    mapping = camera_map or {
        "observation.images.overhead": _DM05_IMAGE_KEYS[0],
        "observation.images.wrist": _DM05_IMAGE_KEYS[1],
    }
    if len(mapping) != 2 or set(mapping.values()) != set(_DM05_IMAGE_KEYS):
        raise ValueError("DM0.5 camera mappings must select images_1 and images_2 exactly once")
    return mapping


def _dm05_stats(path: str) -> dict[str, dict[str, Any]]:
    import torch

    from so101_nexus.policy_adapters.collection import _checkpoint_json

    raw_stats = _checkpoint_json(path, "norm_stats.json")["norm_stats"]
    stats = {}
    for source, target in (("state", "observation.state"), ("action", "action")):
        quantiles = {key: torch.tensor(raw_stats[source][key]) for key in ("q01", "q99")}
        if any(value.shape != (6,) or not value.isfinite().all() for value in quantiles.values()):
            raise ValueError("DM0.5 requires six finite state/action quantiles")
        if not (quantiles["q99"] > quantiles["q01"]).all():
            raise ValueError("DM0.5 requires six increasing state/action quantiles")
        stats[target] = quantiles
    return stats


def load_native_dm05_policy(
    path: str,
    *,
    device: str = "cpu",
    fps: int | None = None,
    camera_map: dict[str, str] | None = None,
) -> LeRobotPolicyAdapter:
    """Load SO101 OpenDM LoRA assets or a converted LeRobot checkpoint locally."""
    try:
        from lerobot.policies.dm05.configuration_dm05 import DM05Config
        from lerobot.policies.dm05.modeling_dm05 import DM05Policy
        from lerobot.policies.dm05.processor_dm05 import (
            DM05TokenizerProcessorStep,
            make_dm05_pre_post_processors,
        )
    except ModuleNotFoundError as exc:
        raise ImportError(
            "Native DM0.5 needs LeRobot's DM05 integration. From a source checkout, run "
            "`uv sync --extra teleop --extra dm05`; for pip setup, see "
            "https://so101-nexus.com/docs/getting-started/installation#native-dm05"
        ) from exc

    import torch
    import yaml
    from lerobot.configs import FeatureType, PolicyFeature
    from lerobot.processor import bind_relative_anchor
    from peft import PeftConfig, PeftModel

    from so101_nexus.config import SO101_JOINT_NAMES
    from so101_nexus.policy_adapters.collection import (
        LeRobotPolicyAdapter,
        _checkpoint_file,
    )
    from so101_nexus.policy_adapters.dm05_processors import DM05SO101TokenizerProcessorStep

    if path != "Dexmal/DM05-SO101-Pick-Cube" and not (Path(path) / "adapter_config.json").is_file():
        return LeRobotPolicyAdapter.from_pretrained(
            path, device=device, policy_type="dm05", fps=fps, camera_map=camera_map
        )
    if fps is None or fps <= 0:
        raise ValueError("DM0.5 requires --policy-fps from its training dataset")
    metadata = yaml.safe_load(_checkpoint_file(path, "meta.yaml").read_text())
    if (
        metadata.get("model_type") != "dm05"
        or metadata.get("robot") != "SO101"
        or metadata.get("checkpoint_type") != "lora"
        or metadata.get("task") != "pick_cube"
    ):
        raise ValueError("Raw DM0.5 collection requires SO101 pick_cube LoRA checkpoint metadata")
    stats = _dm05_stats(path)
    camera_map = _dm05_camera_map(camera_map)
    image_keys = list(_DM05_IMAGE_KEYS)
    config = DM05Config(
        device=device,
        dtype=torch.float32 if device == "cpu" else torch.bfloat16,
        pretrained_name_or_path=metadata["base_model"],
        processor_name_or_path=path,
        chunk_size=50,
        n_action_steps=50,
        use_relative_actions=True,
        action_feature_names=list(SO101_JOINT_NAMES),
        image_keys=image_keys,
        input_features={
            "observation.state": PolicyFeature(FeatureType.STATE, (6,)),
            **{key: PolicyFeature(FeatureType.VISUAL, (3, 448, 448)) for key in image_keys},
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (6,))},
    )
    policy = DM05Policy(config)
    lora_config = PeftConfig.from_pretrained(path)
    # LeRobot's continuous-action core does not expose the raw release's text-generation wrapper.
    lora_config.task_type = None
    policy.model = PeftModel.from_pretrained(
        policy.model,
        path,
        config=lora_config,
        # Transformers 5.10 removed SigLIP's redundant vision_model wrapper.
        key_mapping={r"\.vision_tower\.vision_model\.": ".vision_tower."},
    ).merge_and_unload()
    pre, post = make_dm05_pre_post_processors(config, dataset_stats=stats)
    pre.steps = [
        DM05SO101TokenizerProcessorStep(**step.get_config())
        if isinstance(step, DM05TokenizerProcessorStep)
        else step
        for step in pre.steps
    ]
    bind_relative_anchor(policy, pre)
    policy.to(device).eval()
    return LeRobotPolicyAdapter(policy, pre, post, device=device, fps=fps, camera_map=camera_map)


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
        mapping = _dm05_camera_map(camera_map)
        self.image_keys = [
            next(source for source in mapping if mapping[source] == t) for t in _DM05_IMAGE_KEYS
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
