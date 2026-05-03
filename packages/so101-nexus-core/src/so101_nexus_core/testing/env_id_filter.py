"""Shared assertions for backend env-id filter behavior."""

from __future__ import annotations

from typing import TYPE_CHECKING

from so101_nexus_core.env_ids import all_registered_env_ids, env_ids_for_backend

if TYPE_CHECKING:
    from so101_nexus_core.env_ids import Backend


def run_env_id_filter_contract(
    backend: Backend,
    *,
    prefix: str,
    must_include: list[str],
    min_count: int,
) -> None:
    """Verify env_ids_for_backend(backend) returns the expected slice."""
    ids = env_ids_for_backend(backend)
    assert len(ids) >= min_count, f"expected >= {min_count} ids for {backend}, got {len(ids)}"
    assert all(i.startswith(prefix) for i in ids), (
        f"every id for {backend} must start with {prefix!r}; got {ids}"
    )
    for required in must_include:
        assert required in ids, f"{required!r} missing from {backend} env ids: {ids}"


def assert_none_backend_includes(prefix: str) -> None:
    """env_ids_for_backend(None) returns the full registry; check it includes the prefix."""
    ids = env_ids_for_backend(None)
    assert any(i.startswith(prefix) for i in ids)
    assert ids == all_registered_env_ids()
