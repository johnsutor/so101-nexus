"""MuJoCo backend with lazy Gymnasium environment registration."""

from so101_nexus.env_ids import _register_backend

__all__: list[str] = []

_register_backend("mujoco")
