"""Model adapters for intervention collection using LeRobot observation conventions."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from so101_nexus.config import SO101_JOINT_NAMES

if TYPE_CHECKING:
    from so101_nexus.policy_adapters.chunked_policy import ChunkedActionPolicy


class LeRobotPolicyAdapter:
    """Wrap a LeRobot policy and its saved normalization processor pipelines.

    Parameters
    ----------
    policy
        Loaded LeRobot policy in evaluation mode.
    preprocessor, postprocessor
        Checkpoint processor pipelines, including normalization statistics.
    device
        Torch inference device.
    fps
        Checkpoint action rate, used to validate recording FPS.
    camera_map
        Collector image keys mapped to checkpoint image keys before preprocessing.
    """

    def __init__(
        self,
        policy: Any,
        preprocessor: Any,
        postprocessor: Any,
        *,
        device: str = "cpu",
        fps: int | None = None,
        camera_map: dict[str, str] | None = None,
    ) -> None:
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.device = device
        self.fps = fps
        self.camera_map = camera_map or {}

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        *,
        device: str = "cpu",
        policy_type: str | None = None,
        norm_tag: str | None = None,
        fps: int | None = None,
        camera_map: dict[str, str] | None = None,
    ) -> LeRobotPolicyAdapter:
        """Load a LeRobot checkpoint or an original MolmoAct2 HF checkpoint."""
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        raw_config = _checkpoint_json(path, "config.json")
        original_molmo = "type" not in raw_config and (
            policy_type == "molmoact2" or raw_config.get("model_type") == "molmoact2"
        )
        if original_molmo:
            config = _molmoact2_config(path, norm_tag, device, camera_map or {})
            if path == "allenai/MolmoAct2-SO100_101":
                if fps is not None and fps != 30:
                    raise ValueError("The published MolmoAct2 SO100/101 rollout uses 30 FPS")
                fps = 30
        else:
            config = PreTrainedConfig.from_pretrained(path)
            config.device = device
            if policy_type is not None and config.type != policy_type:
                raise ValueError("--policy-type does not match the checkpoint")
        if config.action_feature is None or config.action_feature.shape != (6,):
            raise ValueError("Collection requires six joint action targets")
        policy_class = get_policy_class(config.type)
        policy = (
            policy_class(config)
            if original_molmo
            else policy_class.from_pretrained(path, config=config)
        )
        pre, post = make_pre_post_processors(
            config,
            pretrained_path=None if original_molmo else path,
            preprocessor_overrides={"device_processor": {"device": device}},
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )
        policy.to(device).eval()
        return cls(policy, pre, post, device=device, fps=fps, camera_map=camera_map)

    def select_action(self, batch: dict[str, Any]) -> Any:
        """Predict one unnormalized joint action from an unbatched observation."""
        import torch
        from lerobot.common.control_utils import predict_action

        observation = {
            key: value
            for key, value in batch.items()
            if key != "task" and key not in self.camera_map and key not in self.camera_map.values()
        }
        observation.update(
            {target: batch[source] for source, target in self.camera_map.items() if source in batch}
        )
        required = getattr(self.policy.config, "input_features", {})
        missing = set(required) - set(observation)
        if missing:
            raise ValueError(
                f"Missing policy observations: {sorted(missing)}; use a custom adapter to map keys"
            )
        for key, feature in required.items():
            shape = feature.shape
            if (
                key.startswith("observation.images.")
                and getattr(self.policy.config, "type", None) == "molmoact2"
            ):
                continue
            if key.startswith("observation.images."):
                shape = (shape[1], shape[2], shape[0])
            if observation[key].shape != shape:
                raise ValueError(
                    f"Policy observation {key!r} requires shape {shape}; "
                    f"got {observation[key].shape}"
                )
        action = predict_action(
            observation,
            self.policy,
            torch.device(self.device),
            self.preprocessor,
            self.postprocessor,
            use_amp=self.policy.config.use_amp,
            task=batch["task"],
            robot_type="sim_so_follower",
        )
        return action.squeeze(0).detach().float().cpu().numpy()

    def reset(self) -> None:
        """Clear model chunks and processor history at episode and handoff boundaries."""
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()


def load_collection_policy(
    *,
    path: str | None = None,
    factory: str | None = None,
    device: str = "cpu",
    policy_type: str | None = None,
    norm_tag: str | None = None,
    fps: int | None = None,
    camera_map: dict[str, str] | None = None,
) -> ChunkedActionPolicy | None:
    """Load a LeRobot checkpoint or call a custom no-argument model factory."""
    if path and factory:
        raise ValueError("Choose either --policy-path or --policy-factory")
    if path:
        return LeRobotPolicyAdapter.from_pretrained(
            path,
            device=device,
            policy_type=policy_type,
            norm_tag=norm_tag,
            fps=fps,
            camera_map=camera_map,
        )
    if not factory:
        return None
    module, separator, name = factory.partition(":")
    if not separator or not module or not name:
        raise ValueError("policy factory must use 'module:function'")
    build = getattr(importlib.import_module(module), name)
    if not callable(build):
        raise TypeError(f"Policy factory {factory!r} is not callable")
    policy = build()
    if not all(callable(getattr(policy, method, None)) for method in ("select_action", "reset")):
        raise TypeError(
            "Policy factory must return an object with select_action(batch) and reset()"
        )
    return cast("ChunkedActionPolicy", policy)


def _checkpoint_json(path: str, filename: str) -> dict[str, Any]:
    from huggingface_hub import hf_hub_download

    location = (
        Path(path) / filename if Path(path).is_dir() else Path(hf_hub_download(path, filename))
    )
    return json.loads(location.read_text())


def _molmoact2_config(path: str, norm_tag: str | None, device: str, camera_map: dict[str, str]):
    from lerobot.configs import FeatureType, PolicyFeature
    from lerobot.policies.factory import make_policy_config

    if norm_tag is None and path == "allenai/MolmoAct2-SO100_101":
        norm_tag = "so100_so101_molmoact2"
    metadata = _checkpoint_json(path, "norm_stats.json").get("metadata_by_tag", {}).get(norm_tag)
    if metadata is None:
        raise ValueError("Original MolmoAct2 checkpoints require a valid --policy-norm-tag")
    for name in ("action_stats", "state_stats"):
        if tuple(metadata[name].get("names", ())) != SO101_JOINT_NAMES:
            raise ValueError("MolmoAct2 normalization must match SO101 joint order")
    if metadata.get("control_mode") != "absolute joint pose":
        raise ValueError("MolmoAct2 collection requires absolute joint pose control")
    image_keys = list(
        metadata.get("camera_keys") or ["observation.images.overhead", "observation.images.side"]
    )
    image_keys = (
        [camera_map.get(key, key) for key in image_keys]
        if metadata.get("camera_keys")
        else list(camera_map.values()) or image_keys
    )
    features = {key: PolicyFeature(FeatureType.VISUAL, (3, 480, 640)) for key in image_keys}
    features["observation.state"] = PolicyFeature(FeatureType.STATE, (6,))
    # The published SO100/101 package uses the pre-PR-777 joint frame.
    frame = (
        {
            "joint_signs": [1.0, -1.0, 1.0, 1.0, 1.0, 1.0],
            "joint_offsets": [0.0, 90.0, 90.0, 0.0, 0.0, 0.0],
        }
        if path == "allenai/MolmoAct2-SO100_101"
        else {}
    )
    return make_policy_config(
        "molmoact2",
        checkpoint_path=path,
        norm_tag=norm_tag,
        device=device,
        inference_action_mode="continuous",
        normalize_gripper=bool(metadata.get("normalize_gripper", False)),
        input_features=features,
        output_features={"action": PolicyFeature(FeatureType.ACTION, (6,))},
        image_keys=image_keys,
        **frame,
    )


def parse_policy_camera_map(mappings: list[str]) -> dict[str, str]:
    """Parse repeatable ``source=target`` camera names without silently colliding."""
    result: dict[str, str] = {}
    for mapping in mappings:
        source, separator, target = mapping.partition("=")
        if not separator or source not in ("wrist", "overhead", "side") or not target:
            raise ValueError("Policy camera mappings must use wrist/overhead/side=target")
        source, target = f"observation.images.{source}", f"observation.images.{target}"
        if source in result or target in result.values():
            raise ValueError("Policy camera mappings must have unique sources and targets")
        result[source] = target
    return result


def validate_policy_fps(
    policy: Any, recording_fps: int, *, expected_fps: int | None = None
) -> None:
    """Require collection FPS to match the checkpoint or an explicit operator setting."""
    model_fps = getattr(policy, "fps", None)
    if model_fps is not None and expected_fps is not None and model_fps != expected_fps:
        raise ValueError(f"Policy requires {model_fps} FPS; --policy-fps specifies {expected_fps}")
    expected_fps = model_fps if model_fps is not None else expected_fps
    if expected_fps is None:
        raise ValueError(
            "Checkpoint has no rate metadata; specify --policy-fps from its training setup"
        )
    if expected_fps <= 0 or recording_fps != expected_fps:
        raise ValueError(f"Set recording FPS to the policy's {expected_fps} FPS")
