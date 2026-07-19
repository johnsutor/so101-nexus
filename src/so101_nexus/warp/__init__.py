"""GPU-batched MuJoCo Warp backend for SO101-Nexus.

Importing this module registers the ``Warp*-v1`` environments. Registration uses
a string ``vector_entry_point``, so this module imports neither torch nor
mujoco_warp; they are pulled only when the env is instantiated. The top-level
``so101_nexus`` package never imports this module (mirroring ``so101_nexus.mujoco``).
"""

from __future__ import annotations

import gymnasium

gymnasium.register(
    id="WarpTouch-v1",
    vector_entry_point="so101_nexus.warp.touch_env:WarpTouchVectorEnv",
    max_episode_steps=512,
)

gymnasium.register(
    id="WarpLookAt-v1",
    vector_entry_point="so101_nexus.warp.look_at_env:WarpLookAtVectorEnv",
    max_episode_steps=256,
)

gymnasium.register(
    id="WarpMove-v1",
    vector_entry_point="so101_nexus.warp.move_env:WarpMoveVectorEnv",
    max_episode_steps=256,
)

gymnasium.register(
    id="WarpPickLift-v1",
    vector_entry_point="so101_nexus.warp.pick_env:WarpPickLiftVectorEnv",
    max_episode_steps=1024,
)

gymnasium.register(
    id="WarpPickAndPlace-v1",
    vector_entry_point="so101_nexus.warp.pick_and_place:WarpPickAndPlaceVectorEnv",
    max_episode_steps=1024,
)

gymnasium.register(
    id="WarpStackCube-v1",
    vector_entry_point="so101_nexus.warp.stack_cube:WarpStackCubeVectorEnv",
    max_episode_steps=1024,
)
