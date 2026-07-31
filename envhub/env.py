"""LeRobot EnvHub entry point for the SO101-Nexus environments.

Loading this file gives the default environment (``MuJoCoPickLift-v1``). Pass an
``EnvConfig`` whose ``task`` names another id, or load the per-environment file
under ``envs/`` to pick one directly:

    make_env("johnsutor/so101-nexus-envs:envs/WarpStackCube-v1.py", ...)

The implementation lives in ``so101_nexus.envhub`` so it is versioned and tested
with the library (see ``requirements.txt``).
"""

from so101_nexus.envhub import make_env

__all__ = ["make_env"]
