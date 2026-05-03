"""Concrete reward shapers (NumPy + torch dual-impl)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from so101_nexus_core.tasks.primitive_target import NumpyContext, TorchContext


class TanhDistanceShaper:
    """Shaper for distance-based metrics: 1 - tanh(scale * distance)."""

    def __init__(self, scale: float):
        self.scale = scale

    def shape_numpy(self, metric: float, ctx: NumpyContext) -> float:  # noqa: D102
        return float(1.0 - np.tanh(self.scale * metric))

    def shape_torch(self, metric, ctx: TorchContext):  # noqa: D102
        import torch

        return 1.0 - torch.tanh(self.scale * metric)


class CosineSimilarityShaper:
    """Shaper for orientation tasks: maps cos_sim in [-1, 1] to progress in [0, 1].

    The metric for orientation tasks is the angular error (radians); we
    convert error -> cos -> normalised progress.
    """

    def shape_numpy(self, metric: float, ctx: NumpyContext) -> float:  # noqa: D102
        cos_sim = float(np.cos(metric))
        return float((cos_sim + 1.0) / 2.0)

    def shape_torch(self, metric, ctx: TorchContext):  # noqa: D102
        import torch

        cos_sim = torch.cos(metric)
        return (cos_sim + 1.0) / 2.0
