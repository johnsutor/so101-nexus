"""Concrete success metrics (NumPy + torch dual-impl)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from so101_nexus_core.tasks.primitive_target import NumpyContext, TorchContext


class DistanceToTarget:
    """Euclidean distance metric. Success when distance < ``success_threshold``."""

    def evaluate_numpy(self, *, target_pos, tcp_pos, tcp_forward, ctx: NumpyContext):  # noqa: D102
        dist = float(np.linalg.norm(target_pos - tcp_pos))
        threshold = ctx.config.success_threshold
        return dist, dist < threshold

    def evaluate_torch(self, *, target_pos, tcp_pos, tcp_forward, ctx: TorchContext):  # noqa: D102
        import torch

        dist = torch.linalg.norm(target_pos - tcp_pos, dim=-1)
        threshold = ctx.config.success_threshold
        return dist, dist < threshold


class OrientationError:
    """Angular error in radians between TCP forward and TCP->target vector.

    Success when error < ``_orientation_success_threshold_rad`` (LookAt's
    private property converting the public degree-valued field).
    """

    def evaluate_numpy(self, *, target_pos, tcp_pos, tcp_forward, ctx: NumpyContext):  # noqa: D102
        assert tcp_forward is not None, "OrientationError requires tcp_forward"
        to_target = target_pos - tcp_pos
        norm = float(np.linalg.norm(to_target))
        if norm > 1e-8:
            to_target = to_target / norm
        cos_sim = float(np.dot(tcp_forward, to_target) / (np.linalg.norm(tcp_forward) + 1e-8))
        cos_sim = float(np.clip(cos_sim, -1.0, 1.0))
        error = float(np.arccos(cos_sim))
        threshold = ctx.config._orientation_success_threshold_rad
        return error, error < threshold

    def evaluate_torch(self, *, target_pos, tcp_pos, tcp_forward, ctx: TorchContext):  # noqa: D102
        import torch

        assert tcp_forward is not None, "OrientationError requires tcp_forward"
        to_target = target_pos - tcp_pos
        norm = torch.linalg.norm(to_target, dim=1, keepdim=True)
        to_target_n = to_target / (norm + 1e-8)
        cos_sim = (tcp_forward * to_target_n).sum(dim=1).clamp(-1, 1)
        error = torch.arccos(cos_sim)
        threshold = ctx.config._orientation_success_threshold_rad
        return error, error < threshold
