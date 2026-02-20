import gymnasium as gym
from mani_skill.utils.wrappers.record import RecordEpisode

import so101_nexus.maniskill  # noqa: F401
from so101_nexus.maniskill.pick_cube import *


def test_load_and_render():
    """Test loading both so100 and so101 environments, rendering, and rolling out."""
    base_kwargs = dict(
        obs_mode="state", num_envs=1, control_mode="pd_joint_delta_pos", render_mode="rgb_array"
    )

    so100_env = gym.make("PickCubeLiftSO101-v1", **base_kwargs)
    so100_env = RecordEpisode(
        so100_env,
        output_dir="videos",
        save_trajectory=True,
        trajectory_name="trajectory",
        save_video=True,
        video_fps=30,
    )

    so100_obs, so100_info = so100_env.reset()

    print(f"SO100 obs shape: {so100_obs.shape}")
    print(f"SO100 action space: {so100_env.action_space}")

    for i in range(100):
        so100_action = so100_env.action_space.sample()

        so100_result = so100_env.step(so100_action)

        so100_obs, so100_reward, so100_terminated, so100_truncated, so100_info = so100_result

        print(f"Step {i}: SO100 reward={so100_reward}")

    so100_env.close()
    print("Test completed successfully! Videos saved to 'videos/' directory.")


if __name__ == "__main__":
    test_load_and_render()
