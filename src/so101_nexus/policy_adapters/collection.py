"""Model adapters for intervention collection using LeRobot observation conventions."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, cast

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
    """

    def __init__(
        self, policy: Any, preprocessor: Any, postprocessor: Any, *, device: str = "cpu"
    ) -> None:
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.device = device

    @classmethod
    def from_pretrained(cls, path: str, *, device: str = "cpu") -> LeRobotPolicyAdapter:
        """Load a local or Hub checkpoint with its saved processor pipelines."""
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        config = PreTrainedConfig.from_pretrained(path)
        config.device = device
        if config.action_feature is None or config.action_feature.shape != (6,):
            raise ValueError(
                "Collection requires six joint action targets; use a custom adapter otherwise"
            )
        policy = get_policy_class(config.type).from_pretrained(path, config=config)
        policy.to(device).eval()
        pre, post = make_pre_post_processors(
            config,
            pretrained_path=path,
            preprocessor_overrides={"device_processor": {"device": device}},
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )
        return cls(policy, pre, post, device=device)

    def select_action(self, batch: dict[str, Any]) -> Any:
        """Predict one unnormalized joint action from an unbatched observation."""
        import torch
        from lerobot.utils.control_utils import predict_action

        observation = {key: value for key, value in batch.items() if key != "task"}
        required = getattr(self.policy.config, "input_features", {})
        missing = set(required) - set(observation)
        if missing:
            raise ValueError(
                f"Missing policy observations: {sorted(missing)}; use a custom adapter to map keys"
            )
        for key, feature in required.items():
            shape = feature.shape
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
    *, path: str | None = None, factory: str | None = None, device: str = "cpu"
) -> ChunkedActionPolicy | None:
    """Load a LeRobot checkpoint or call a custom no-argument model factory."""
    if path and factory:
        raise ValueError("Choose either --policy-path or --policy-factory")
    if path:
        return LeRobotPolicyAdapter.from_pretrained(path, device=device)
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
