import gymnasium

gymnasium.register(
    id="MuJoCoPickCubeGoal-v1",
    entry_point="so101_nexus_mujoco.pick_cube:PickCubeEnv",
    max_episode_steps=256,
)

gymnasium.register(
    id="MuJoCoPickCubeLift-v1",
    entry_point="so101_nexus_mujoco.pick_cube:PickCubeLiftEnv",
    max_episode_steps=256,
)

gymnasium.register(
    id="MuJoCoPickAndPlace-v1",
    entry_point="so101_nexus_mujoco.pick_and_place:PickAndPlaceEnv",
    max_episode_steps=256,
)
