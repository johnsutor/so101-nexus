"""Rendering quality reaches every native batch world."""

import mujoco
import numpy as np
import pytest

from so101_nexus import OverheadCamera, RenderConfig, RenderingQuality, TouchConfig, WristCamera

pytest.importorskip("mjbatch")


def test_reproducible_quality_and_images_in_every_world(env_factory):
    config = TouchConfig(
        render=RenderConfig(width=64, height=48, quality=RenderingQuality(profile="reproducible")),
        observations=[WristCamera(width=64, height=48), OverheadCamera(width=64, height=48)],
    )
    batch = env_factory(backend="mjbatch", config=config, render_mode="rgb_array", num_threads=1)
    reference = env_factory(config=config, render_mode="rgb_array")
    expected, _ = reference.reset(seed=17)
    actual, _ = batch.reset(seed=[17, 17])
    for world, env in enumerate(batch.envs):
        base = env.unwrapped
        assert base.model.vis.quality.offsamples == 4
        for renderer in (base._wrist_renderer, base._overhead_obs_renderer):
            assert not renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]
            assert renderer._mjr_context.offSamples == 4
        for name, image in expected.items():
            np.testing.assert_array_equal(actual[name][world], image)
        np.testing.assert_array_equal(env.render(), reference.render())
        assert not base._renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]
    for _ in range(3):
        action = np.zeros(batch.action_space.shape, dtype=np.float32)
        actual, reward, terminated, truncated, _ = batch.step(action)
        expected, expected_reward, expected_terminated, expected_truncated, _ = reference.step(
            action[0]
        )
        for world in range(batch.num_envs):
            for name, image in expected.items():
                np.testing.assert_array_equal(actual[name][world], image)
            assert reward[world] == pytest.approx(expected_reward, abs=1e-7)
            assert terminated[world] == expected_terminated
            assert truncated[world] == expected_truncated
