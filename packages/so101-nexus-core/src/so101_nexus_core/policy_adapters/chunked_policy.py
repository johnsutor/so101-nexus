"""Protocol for chunked-action policies."""

from __future__ import annotations

from typing import Any, Protocol


class ChunkedActionPolicy(Protocol):
    """Minimal LeRobot-shaped protocol for policies with internal chunk replay."""

    def select_action(self, batch: dict[str, Any]) -> Any:
        """Return one action for the current observation batch."""

    def reset(self) -> None:
        """Clear cached state at an episode boundary."""
