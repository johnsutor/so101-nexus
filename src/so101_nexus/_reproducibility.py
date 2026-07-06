"""Small reproducibility helpers shared by training scripts."""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    """Seed Python, NumPy, torch, and torch device RNGs.

    Mirrors the useful subset of ``accelerate.utils.set_seed`` without making the
    trainers depend on Accelerate. ``deterministic=True`` asks PyTorch for deterministic
    kernels where available and uses warn-only mode so backend-specific nondeterministic
    kernels do not abort long-running Warp training jobs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
