"""Tests for SO101 action processor steps."""

from __future__ import annotations


def test_processors_subpackage_imports() -> None:
    """The subpackage and module stubs are importable on a base install."""
    import so101_nexus_core.processors  # noqa: F401
    from so101_nexus_core.processors import action, observation, pipelines  # noqa: F401
