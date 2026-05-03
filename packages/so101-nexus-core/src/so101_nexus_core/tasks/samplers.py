"""Concrete target samplers (NumPy + torch dual-impl)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from so101_nexus_core.tasks.primitive_target import NumpyContext, TorchContext


class WorkspaceTargetSampler:
    """Reach: sample a target position in an annular sector on the floor.

    Uses ``config.spawn_center``, ``spawn_min_radius``, ``spawn_max_radius``,
    and ``spawn_angle_half_range_deg``. Z is fixed at ``config.target_radius``
    to keep the marker visible above the floor.
    """

    def sample_numpy(self, ctx: NumpyContext) -> np.ndarray:  # noqa: D102
        cfg = ctx.config
        cx, cy = cfg.spawn_center
        min_r = cfg.spawn_min_radius
        max_r = cfg.spawn_max_radius
        angle_half = float(np.radians(cfg.spawn_angle_half_range_deg))

        r = ctx.rng.uniform(min_r, max_r)
        theta = ctx.rng.uniform(-angle_half, angle_half)
        x = cx + r * np.cos(theta)
        y = cy + r * np.sin(theta)
        z = cfg.target_radius

        return np.array([x, y, z], dtype=np.float64)

    def sample_torch(self, ctx: TorchContext):  # noqa: D102
        import torch

        cfg = ctx.config
        cx, cy = cfg.spawn_center
        min_r = cfg.spawn_min_radius
        max_r = cfg.spawn_max_radius
        angle_half = float(np.radians(cfg.spawn_angle_half_range_deg))
        b = ctx.batch_size

        r = torch.rand(b, device=ctx.device) * (max_r - min_r) + min_r
        theta = (torch.rand(b, device=ctx.device) * 2 - 1) * angle_half
        x = cx + r * torch.cos(theta)
        y = cy + r * torch.sin(theta)
        z = torch.full((b,), cfg.target_radius, device=ctx.device)
        return torch.stack([x, y, z], dim=1)


class TcpRelativeSampler:
    """Move: sample a target by translating the current TCP by a fixed direction*distance."""

    def __init__(
        self,
        direction_vec_attr: str = "direction",
        distance_attr: str = "target_distance",
    ):
        self.direction_vec_attr = direction_vec_attr
        self.distance_attr = distance_attr

    def sample_numpy(self, ctx: NumpyContext) -> np.ndarray:  # noqa: D102
        from so101_nexus_core.config._types import DIRECTION_VECTORS

        assert ctx.tcp_pos is not None, "TcpRelativeSampler.sample_numpy requires ctx.tcp_pos"
        cfg = ctx.config
        direction = getattr(cfg, self.direction_vec_attr)
        distance = getattr(cfg, self.distance_attr)
        dir_vec = np.array(DIRECTION_VECTORS[direction], dtype=np.float64)
        target = ctx.tcp_pos + dir_vec * distance
        target[2] = max(target[2], 0.02)
        return target.astype(np.float64)

    def sample_torch(self, ctx: TorchContext):  # noqa: D102
        import torch

        from so101_nexus_core.config._types import DIRECTION_VECTORS

        assert ctx.tcp_pos is not None, "TcpRelativeSampler.sample_torch requires ctx.tcp_pos"
        cfg = ctx.config
        direction = getattr(cfg, self.direction_vec_attr)
        distance = getattr(cfg, self.distance_attr)
        dir_vec = torch.tensor(DIRECTION_VECTORS[direction], device=ctx.device, dtype=torch.float32)
        target = ctx.tcp_pos + dir_vec * distance
        target[:, 2] = target[:, 2].clamp(min=0.02)
        return target


class FloorObjectSampler:
    """LookAt: sample a 2D position on the floor inside a square spawn region.

    Uses ``config.spawn_center`` and ``config.spawn_half_size``. Z is set to the
    object's ``half_size`` (the marker rests on the floor).
    """

    def sample_numpy(self, ctx: NumpyContext) -> np.ndarray:  # noqa: D102
        cfg = ctx.config
        half = cfg.spawn_half_size
        cx, cy = cfg.spawn_center
        x = cx + ctx.rng.uniform(-half, half)
        y = cy + ctx.rng.uniform(-half, half)
        target_obj = cfg.objects[0]
        z = target_obj.half_size
        return np.array([x, y, z], dtype=np.float64)

    def sample_torch(self, ctx: TorchContext):  # noqa: D102
        import torch

        cfg = ctx.config
        half = cfg.spawn_half_size
        cx, cy = cfg.spawn_center
        b = ctx.batch_size
        x = cx + (torch.rand(b, device=ctx.device) * 2 - 1) * half
        y = cy + (torch.rand(b, device=ctx.device) * 2 - 1) * half
        target_obj = cfg.objects[0]
        z = torch.full((b,), target_obj.half_size, device=ctx.device)
        return torch.stack([x, y, z], dim=1)
