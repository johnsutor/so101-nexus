"""Observation-layout helpers shared across simulation-backend test suites."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from so101_nexus.observations import Observation


def component_slice(env: Any, component: type[Observation]) -> slice:
    """Return the flat-observation slice occupied by an observation component.

    Locating a component by type rather than by a hardcoded offset keeps tests
    readable and stops them from silently checking the wrong columns whenever a
    component is added to a task's default ``observations`` list.

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
    AssertionError
        If the env's observation list contains no instance of ``component``.
    """
    offset = 0
    for comp in env.unwrapped.config.observations:
        if isinstance(comp, component):
            return slice(offset, offset + comp.size)
        offset += comp.size
    raise AssertionError(f"{component.__name__} is not in the env's observation list")
