"""GPU-batched MuJoCo Warp backend for SO101-Nexus.

Importing this module registers the ``Warp*-v1`` environments. Registration uses
a string ``vector_entry_point``, so this module imports neither torch nor
mujoco_warp; they are pulled only when the env is instantiated. The top-level
``so101_nexus`` package never imports this module (mirroring ``so101_nexus.mujoco``).
"""

from __future__ import annotations

import gymnasium

gymnasium.register(
    id="WarpReach-v1",
    vector_entry_point="so101_nexus.warp.reach_env:WarpReachVectorEnv",
    max_episode_steps=512,
)
