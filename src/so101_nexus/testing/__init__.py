"""Reusable pytest helpers for SO101-Nexus environments.

This subpackage is intentionally shipped as part of ``so101_nexus``
(not under ``tests/``) so backend packages can import the shared
Gymnasium-contract suite from their own test modules.
"""

from __future__ import annotations

from so101_nexus.testing.contract import run_env_contract
from so101_nexus.testing.observations import component_slice

__all__ = ["component_slice", "run_env_contract"]
