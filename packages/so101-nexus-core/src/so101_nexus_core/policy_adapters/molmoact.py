"""MolmoAct2 chunked policy adapter."""

from __future__ import annotations

import contextlib
import importlib
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence


class MolmoActPolicy:
    """Chunked-policy adapter around ``allenai/MolmoAct2-SO100_101``.

    Parameters
    ----------
    model : Any
        Model object exposing ``predict_action``.
    processor : Any
        Processor passed through to ``model.predict_action``.
    image_keys : Sequence[str], optional
        LeRobot batch image keys, in the order passed to the model. Defaults to
        overhead then wrist. When used with ``RolloutRecorder``, keep these keys
        aligned with the recorder's ``camera_keys``.
    state_key : str, optional
        LeRobot batch key for the degree-valued robot state.
    task_key : str, optional
        LeRobot batch key for the task string.
    norm_tag : str, optional
        Normalization tag forwarded to MolmoAct2.
    action_mode : {"continuous", "discrete"}, optional
        MolmoAct2 action mode. Discrete mode requires ``action_tokenizer``.
    chunk_size : int, optional
        Number of returned actions replayed before querying the model again.
    num_steps : int, optional
        Flow solver step count forwarded to MolmoAct2.
    normalize_language : bool, optional
        Whether MolmoAct2 should normalize the task text.
    enable_depth_reasoning : bool, optional
        Whether to enable MolmoAct2 depth reasoning.
    enable_cuda_graph : bool, optional
        Whether to enable MolmoAct2 CUDA graph capture.
    action_tokenizer : Any, optional
        Tokenizer required by discrete action mode.

    Notes
    -----
    ``select_action`` returns absolute joint positions in LeRobot degree units.
    The rollout recorder converts those actions to radians for ``env.step``.
    If real-model rollouts drift or immediately saturate the action clip, check
    whether the checkpoint is emitting deltas instead of absolute positions.
    """

    def __init__(
        self,
        model: Any,
        processor: Any,
        *,
        image_keys: Sequence[str] = (
            "observation.images.overhead",
            "observation.images.wrist",
        ),
        state_key: str = "observation.state",
        task_key: str = "task",
        norm_tag: str = "so100_so101_molmoact2",
        action_mode: Literal["continuous", "discrete"] = "continuous",
        chunk_size: int = 8,
        num_steps: int = 10,
        normalize_language: bool = True,
        enable_depth_reasoning: bool = False,
        enable_cuda_graph: bool = True,
        action_tokenizer: Any | None = None,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        if num_steps <= 0:
            raise ValueError("num_steps must be positive.")
        if action_mode == "discrete" and action_tokenizer is None:
            raise ValueError("action_tokenizer is required when action_mode='discrete'.")

        self.model = model
        self.processor = processor
        self.image_keys = tuple(image_keys)
        self.state_key = state_key
        self.task_key = task_key
        self.norm_tag = norm_tag
        self.action_mode = action_mode
        self.chunk_size = chunk_size
        self.num_steps = num_steps
        self.normalize_language = normalize_language
        self.enable_depth_reasoning = enable_depth_reasoning
        self.enable_cuda_graph = enable_cuda_graph
        self.action_tokenizer = action_tokenizer
        self._action_queue: list[np.ndarray] = []

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str = "allenai/MolmoAct2-SO100_101",
        *,
        device: Any | None = None,
        dtype: Any | None = None,
        **policy_kwargs: Any,
    ) -> MolmoActPolicy:
        """Load the MolmoAct2 checkpoint and wrap it in ``MolmoActPolicy``."""
        try:
            from huggingface_hub import snapshot_download

            transformers = importlib.import_module("transformers")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Install the 'molmoact' extra before calling MolmoActPolicy.from_pretrained."
            ) from exc

        model_path = snapshot_download(repo_id)
        processor = transformers.AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model_kwargs: dict[str, Any] = {"trust_remote_code": True}
        if dtype is not None:
            model_kwargs["dtype"] = dtype
        if device is not None:
            model_kwargs["device_map"] = device
        model = transformers.AutoModelForImageTextToText.from_pretrained(
            model_path,
            **model_kwargs,
        ).eval()
        return cls(model, processor, **policy_kwargs)

    def select_action(self, batch: dict[str, Any]) -> np.ndarray:
        """Return one 6-dim action in degrees for the current observation batch."""
        if not self._action_queue:
            self._refill_action_queue(batch)
        return self._action_queue.pop(0)

    def reset(self) -> None:
        """Clear any cached action chunk."""
        self._action_queue.clear()

    def _refill_action_queue(self, batch: dict[str, Any]) -> None:
        images = [batch[key] for key in self.image_keys]
        state = np.asarray(batch[self.state_key], dtype=np.float32)
        kwargs: dict[str, Any] = {
            "processor": self.processor,
            "images": images,
            "task": batch[self.task_key],
            "state": state,
            "norm_tag": self.norm_tag,
            "action_mode": self.action_mode,
            "enable_depth_reasoning": self.enable_depth_reasoning,
            "num_steps": self.num_steps,
            "normalize_language": self.normalize_language,
            "enable_cuda_graph": self.enable_cuda_graph,
        }
        if self.action_tokenizer is not None:
            kwargs["action_tokenizer"] = self.action_tokenizer

        with _inference_mode():
            output = self.model.predict_action(**kwargs)

        actions = _to_numpy(output.actions)
        chunk = np.asarray(actions[0, : self.chunk_size], dtype=np.float32)
        if chunk.ndim != 2 or chunk.shape[1] != 6:
            raise ValueError(
                "MolmoAct2 predict_action must return actions with shape "
                f"(batch, steps, 6); got {actions.shape}."
            )
        if chunk.shape[0] == 0:
            raise ValueError("MolmoAct2 predict_action returned an empty action chunk.")
        self._action_queue.extend(np.asarray(row, dtype=np.float32) for row in chunk)


def _to_numpy(value: Any) -> np.ndarray:
    """Convert a torch-like tensor or array-like value to ``np.ndarray``."""
    if hasattr(value, "float") and hasattr(value, "cpu") and hasattr(value, "numpy"):
        return value.float().cpu().numpy()
    return np.asarray(value)


def _inference_mode() -> contextlib.AbstractContextManager[None]:
    """Return ``torch.inference_mode`` when torch is installed."""
    try:
        import torch
    except ModuleNotFoundError:
        return contextlib.nullcontext()
    return torch.inference_mode()
