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

# --- YCB generic environments ---
gymnasium.register(
    id="MuJoCoPickYCBGoal-v1",
    entry_point="so101_nexus_mujoco.pick_ycb:PickYCBEnv",
    max_episode_steps=256,
)

gymnasium.register(
    id="MuJoCoPickYCBLift-v1",
    entry_point="so101_nexus_mujoco.pick_ycb:PickYCBLiftEnv",
    max_episode_steps=256,
)

# --- YCB per-object environments ---
_YCB_ENV_NAME_MAP = {
    "009_gelatin_box": "GelatinBox",
    "011_banana": "Banana",
    "030_fork": "Fork",
    "031_spoon": "Spoon",
    "032_knife": "Knife",
    "033_spatula": "Spatula",
    "037_scissors": "Scissors",
    "040_large_marker": "LargeMarker",
    "043_phillips_screwdriver": "PhillipsScrewdriver",
    "058_golf_ball": "GolfBall",
}

for _model_id, _env_name in _YCB_ENV_NAME_MAP.items():
    gymnasium.register(
        id=f"MuJoCoPick{_env_name}Goal-v1",
        entry_point="so101_nexus_mujoco.pick_ycb:PickYCBEnv",
        max_episode_steps=256,
        kwargs={"model_id": _model_id},
    )
    gymnasium.register(
        id=f"MuJoCoPick{_env_name}Lift-v1",
        entry_point="so101_nexus_mujoco.pick_ycb:PickYCBLiftEnv",
        max_episode_steps=256,
        kwargs={"model_id": _model_id},
    )
