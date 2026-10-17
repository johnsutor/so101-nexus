"""GPU-batched Warp backend with lazy Gymnasium vector registration."""

from so101_nexus.env_ids import _register_backend

__all__: list[str] = []

_register_backend("warp")
