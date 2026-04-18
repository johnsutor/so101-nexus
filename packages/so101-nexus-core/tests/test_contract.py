"""Smoke tests that `run_env_contract` is importable and has the right shape."""

from __future__ import annotations

import inspect


def test_contract_importable():
    from so101_nexus_core.testing import run_env_contract

    assert callable(run_env_contract)


def test_contract_signature():
    from so101_nexus_core.testing import run_env_contract

    sig = inspect.signature(run_env_contract)
    assert "env_id" in sig.parameters
    assert "config_cls" in sig.parameters
    assert "reward_range" in sig.parameters
