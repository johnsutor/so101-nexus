"""Backend-agnostic spec for primitive target tasks (Reach, Move, LookAt).

Concrete sampler/metric/shaper protocols expose both a NumPy and torch
implementation. The NumPy path is consumed by the MuJoCo backend (single-
env, np.random.Generator). The torch path is consumed by the ManiSkill
backend (vectorized, device tensors). A future MuJoCo Warp backend will
plug into the torch path via Warp/torch DLPack interop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np
    import torch

    from so101_nexus_core.observations import Observation


@dataclass
class NumpyContext:
    """Per-call context handed to NumPy spec implementations (single-env)."""

    rng: np.random.Generator
    config: Any
    tcp_pos: np.ndarray | None = None


@dataclass
class TorchContext:
    """Per-call context handed to torch spec implementations (batched)."""

    rng: torch.Generator | None
    device: Any
    config: Any
    batch_size: int
    tcp_pos: torch.Tensor | None = None


@dataclass(frozen=True)
class TargetMarker:
    """Visual marker for a primitive target task.

    ``is_kinematic=True`` means the marker is a visual-only site/actor with no
    physics. ``is_kinematic=False`` means a dynamic body (e.g., LookAt's cube)
    with a freejoint and collision geom.
    """

    name: str
    shape: Literal["sphere", "cube"]
    size: float
    rgba: tuple[float, float, float, float]
    is_kinematic: bool = True
    mass: float = 0.01
    color_name: str | None = None


class TargetSampler(Protocol):
    """Sample a target world position. Both impls must be defined."""

    def sample_numpy(self, ctx: NumpyContext) -> np.ndarray: ...  # noqa: D102
    def sample_torch(self, ctx: TorchContext) -> torch.Tensor: ...  # noqa: D102


class SuccessMetric(Protocol):
    """Compute the scalar metric and success flag from current TCP state + target."""

    def evaluate_numpy(  # noqa: D102
        self,
        *,
        target_pos: np.ndarray,
        tcp_pos: np.ndarray,
        tcp_forward: np.ndarray | None,
        ctx: NumpyContext,
    ) -> tuple[float, bool]: ...
    def evaluate_torch(  # noqa: D102
        self,
        *,
        target_pos: torch.Tensor,
        tcp_pos: torch.Tensor,
        tcp_forward: torch.Tensor | None,
        ctx: TorchContext,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


class RewardShaper(Protocol):
    """Convert metric -> progress in [0, 1]."""

    def shape_numpy(self, metric: float, ctx: NumpyContext) -> float: ...  # noqa: D102
    def shape_torch(self, metric: torch.Tensor, ctx: TorchContext) -> torch.Tensor: ...  # noqa: D102


@dataclass(frozen=True)
class PrimitiveTargetSpec:
    """Composable specification for a primitive target task.

    ``task_description`` is either a static string or a callable consuming the
    config and returning the rendered string.

    ``requires_tcp_pos_for_sampling`` is True for Move-style samplers that
    need forward kinematics first.

    ``requires_tcp_forward_for_metric`` is True for orientation tasks (LookAt).
    """

    marker: TargetMarker
    sampler: TargetSampler
    metric: SuccessMetric
    shaper: RewardShaper
    obs_extra: type[Observation] | None
    task_description: str | Callable[[Any], str]
    requires_tcp_pos_for_sampling: bool = False
    requires_tcp_forward_for_metric: bool = False
    success_threshold_attr: str = "success_threshold"


def resolve_task_description(spec: PrimitiveTargetSpec, config: Any) -> str:
    """Resolve a static or callable task_description against *config*."""
    td = spec.task_description
    return td(config) if callable(td) else td
