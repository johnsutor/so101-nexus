"""LeRobot EnvHub entry point for the MuJoCoPickAndPlace-v2 environment."""

from functools import partial

from so101_nexus.envhub import make_env as _make_env

make_env = partial(_make_env, env_id="MuJoCoPickAndPlace-v2")

__all__ = ["make_env"]
