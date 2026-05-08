"""Shared env-id filter test contracts for backend packages."""

from __future__ import annotations

from typing import TYPE_CHECKING

from so101_nexus_core.env_ids import all_registered_env_ids, env_ids_for_backend

if TYPE_CHECKING:
    from collections.abc import Sequence

    from so101_nexus_core.env_ids import Backend


def run_env_id_filter_contract(
    *,
    backend: Backend,
    prefix: str,
    min_count: int,
    required_ids: Sequence[str],
) -> None:
    """Verify backend filtering returns the expected registered env ids."""
    ids = env_ids_for_backend(backend)

    assert len(ids) >= min_count
    assert all(env_id.startswith(prefix) for env_id in ids)
    for env_id in required_ids:
        assert env_id in ids

    all_ids = env_ids_for_backend(None)
    assert any(env_id.startswith(prefix) for env_id in all_ids)
    assert all_ids == all_registered_env_ids()
