"""Observation-layout helpers shared across simulation-backend test suites."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from so101_nexus.observations import component_slice as _component_slice

if TYPE_CHECKING:
    from so101_nexus.observations import Observation


def component_slice(env: Any, component: type[Observation]) -> slice:
    """Return the flat-observation slice occupied by an observation component.

    Thin ``env``-shaped wrapper over :func:`so101_nexus.observations.component_slice`
    so tests can name an env instead of digging its component list out of its
    config.

    Parameters
    ----------
    env
        A constructed environment (wrapped or not) exposing
        ``unwrapped.config.observations``.
    component
        The component class to locate.

    Returns
    -------
    slice
        Start/stop indices into the flat state vector. For a batched Warp env
        this indexes the trailing (feature) axis.

    Raises
    ------
    ValueError
        If the env's observation list contains no instance of ``component``.
    """
    return _component_slice(env.unwrapped.config.observations, component)
