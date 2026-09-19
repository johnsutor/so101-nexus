"""Reproducible training RNGs and deterministic execution modes."""

import random

import numpy as np
import pytest
import torch

from so101_nexus._reproducibility import seed_everything


@pytest.fixture(autouse=True)
def restore_rng_settings(monkeypatch):
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    flags = (
        (torch.backends.cudnn, "benchmark"),
        (torch.backends.cudnn, "deterministic"),
        (torch.backends.cudnn, "allow_tf32"),
        (torch.backends.cuda.matmul, "allow_tf32"),
    )
    previous = [(owner, name, getattr(owner, name)) for owner, name in flags]
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    yield
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(torch_state)
    torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
    for owner, name, value in previous:
        setattr(owner, name, value)


def test_seed_everything_replays_all_cpu_rngs():
    def draw(seed):
        seed_everything(seed, deterministic=True)
        return random.random(), np.random.random(), torch.rand(4)

    first, repeated, other = draw(17), draw(17), draw(18)
    assert first[:2] == repeated[:2]
    assert torch.equal(first[2], repeated[2])
    assert first[:2] != other[:2]
    assert not torch.equal(first[2], other[2])


@pytest.mark.parametrize("warn_only", [False, True])
def test_deterministic_mode_is_strict_unless_explicitly_relaxed(warn_only):
    seed_everything(0, deterministic=True, deterministic_warn_only=warn_only)
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.is_deterministic_algorithms_warn_only_enabled() is warn_only
    assert not torch.backends.cudnn.benchmark
    assert not torch.backends.cuda.matmul.allow_tf32


def test_invalid_cublas_config_fails_before_training(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "invalid")
    with pytest.raises(ValueError, match="CUBLAS_WORKSPACE_CONFIG"):
        seed_everything(0, deterministic=True)


def test_late_cuda_configuration_is_rejected(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG")
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    with pytest.raises(RuntimeError, match="before CUDA"):
        seed_everything(0, deterministic=True)
