"""Reusable pytest helpers for SO101-Nexus environments.

This subpackage is intentionally shipped as part of ``so101_nexus_core``
(not under ``tests/``) so backend packages can import the shared
Gymnasium-contract suite from their own test modules.
"""

from __future__ import annotations

from so101_nexus_core.testing.contract import run_env_contract
from so101_nexus_core.testing.gpu import skip_if_vectorized_runtime_unavailable

__all__ = ["run_env_contract", "skip_if_vectorized_runtime_unavailable"]
