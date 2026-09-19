"""Small reproducibility helpers shared by training scripts."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(
    seed: int, *, deterministic: bool = False, deterministic_warn_only: bool = False
) -> None:
    """Seed Python, NumPy, torch, and torch device RNGs.

    Mirrors the useful subset of ``accelerate.utils.set_seed`` without making the
    trainers depend on Accelerate. ``deterministic=True`` rejects known nondeterministic
    Torch operations unless ``deterministic_warn_only`` explicitly permits them.
    This setting does not control external simulator kernels.
    """
    if deterministic:
        workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if workspace is None and torch.cuda.is_initialized():
            raise RuntimeError("Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA initialization.")
        if workspace not in (None, ":4096:8", ":16:8"):
            raise ValueError("CUBLAS_WORKSPACE_CONFIG must be :4096:8 or :16:8.")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(deterministic, warn_only=deterministic_warn_only)
