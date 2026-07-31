"""LeRobot EnvHub entry point for the MuJoCoPickLift-v1 environment.

The implementation lives in ``so101_nexus.envhub`` so it is versioned and tested
with the library (see ``requirements.txt``).
"""

from functools import partial

from so101_nexus.envhub import make_env as _make_env

make_env = partial(_make_env, env_id="MuJoCoPickLift-v1")

__all__ = ["make_env"]
