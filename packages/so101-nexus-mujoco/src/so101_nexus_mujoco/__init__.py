"""MuJoCo backend for SO101-Nexus: registers all MuJoCo Gymnasium environments."""

import gymnasium

gymnasium.register(
    id="MuJoCoPickLift-v1",
    entry_point="so101_nexus_mujoco.pick_env:PickLiftEnv",
    max_episode_steps=1024,
)

gymnasium.register(
    id="MuJoCoPickAndPlace-v1",
    entry_point="so101_nexus_mujoco.pick_and_place:PickAndPlaceEnv",
    max_episode_steps=1024,
)

gymnasium.register(
    id="MuJoCoReach-v1",
    entry_point="so101_nexus_mujoco.reach_env:ReachEnv",
    max_episode_steps=512,
)

gymnasium.register(
    id="MuJoCoLookAt-v1",
    entry_point="so101_nexus_mujoco.look_at_env:LookAtEnv",
    max_episode_steps=256,
)

gymnasium.register(
    id="MuJoCoMove-v1",
    entry_point="so101_nexus_mujoco.move_env:MoveEnv",
    max_episode_steps=256,
)
