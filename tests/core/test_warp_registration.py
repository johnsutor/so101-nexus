"""WarpReach-v1 registration + discovery, proven without the warp extra."""


def test_warp_reach_registered_and_discoverable():
    import gymnasium as gym

    import so101_nexus.warp  # noqa: F401  (import-light: string vector_entry_point)
    from so101_nexus.env_ids import env_ids_for_backend

    assert "WarpReach-v1" in gym.envs.registry
    spec = gym.envs.registry["WarpReach-v1"]
    assert spec.max_episode_steps == 512
    assert "WarpReach-v1" in env_ids_for_backend("warp")
    assert "WarpReach-v1" not in env_ids_for_backend("mujoco")
