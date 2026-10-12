"""Rendering quality allocation and local full-state repeatability."""

import json

import mujoco
import numpy as np
import pytest

from so101_nexus import OverheadCamera, RenderConfig, RenderingQuality, TouchConfig, WristCamera


def _config(quality):
    return TouchConfig(
        render=RenderConfig(width=64, height=48, camera="side", quality=quality),
        observations=[WristCamera(width=64, height=48), OverheadCamera(width=64, height=48)],
    )


@pytest.mark.parametrize(
    "profile,samples,shadows", [("default", 8, True), ("reproducible", 4, False)]
)
def test_quality_applied_before_all_camera_allocations(env_factory, profile, samples, shadows):
    env = env_factory(config=_config(RenderingQuality(profile=profile)), render_mode="rgb_array")
    base = env.unwrapped
    assert base.model.vis.quality.offsamples == samples
    assert base.model.vis.quality.shadowsize == 8192
    assert "render" not in base.rendering_metadata["renderers"]
    env.reset(seed=17)
    env.render()
    for name, renderer in (
        ("wrist", base._wrist_renderer),
        ("overhead", base._overhead_obs_renderer),
        ("render", base._renderer),
    ):
        assert bool(renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]) is shadows
        metadata = base.rendering_metadata["renderers"][name]
        assert metadata["requested_msaa_samples"] == samples
        if profile == "reproducible":
            assert metadata["actual_msaa_samples"] == 4
        assert metadata["actual_msaa_samples"] == renderer._mjr_context.offSamples
        assert metadata["shadows"] is shadows
        assert (metadata["width"], metadata["height"]) == (64, 48)
    json.dumps(base.rendering_metadata)


@pytest.mark.parametrize("field,value", [("msaa_samples", 0), ("shadow_map_size", 1024)])
def test_camera_model_quality_conflict_fails_before_allocation(env_factory, field, value):
    config = _config(RenderingQuality(profile="reproducible"))
    config.observations[0].quality = RenderingQuality(profile="reproducible", **{field: value})
    with pytest.raises(ValueError, match=field):
        env_factory(config=config)


def test_per_camera_shadow_override_and_runtime_quality_rejection(env_factory):
    config = _config(RenderingQuality(profile="reproducible"))
    config.observations[0].quality = RenderingQuality(profile="reproducible", shadows=True)
    env = env_factory(config=config, render_mode="rgb_array")
    assert env.unwrapped._wrist_renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]
    env.reset(seed=17)
    config.render.quality = RenderingQuality()
    with pytest.raises(RuntimeError, match=r"quality.*construction"):
        env.render()
    assert env.unwrapped.model.vis.quality.offsamples == 4


def _images(env):
    obs = env.unwrapped._get_obs()
    return {"wrist": obs["wrist_camera"], "overhead": obs["overhead_camera"], "side": env.render()}


def test_reproducible_profile_restored_full_state_across_resets_and_contexts(env_factory):
    reference = env_factory(
        config=_config(RenderingQuality(profile="reproducible")), render_mode="rgb_array"
    )
    reference.reset(seed=31)
    reference.step(np.zeros(reference.action_space.shape, dtype=np.float32))
    base = reference.unwrapped
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(base.model, spec))
    mujoco.mj_getState(base.model, base.data, state, spec)
    model_state = {
        name: getattr(base.model, name).copy()
        for name in (
            "cam_pos",
            "cam_quat",
            "cam_fovy",
            "geom_rgba",
            "mat_rgba",
            "body_pos",
            "site_pos",
        )
    }
    mujoco.mj_forward(base.model, base.data)
    expected = _images(reference)
    for new_context in (False, True):
        env = (
            env_factory(
                config=_config(RenderingQuality(profile="reproducible")), render_mode="rgb_array"
            )
            if new_context
            else reference
        )
        for seed in (31, 47):
            env.reset(seed=seed)
            inner = env.unwrapped
            for name, value in model_state.items():
                getattr(inner.model, name)[:] = value
            mujoco.mj_setState(inner.model, inner.data, state, spec)
            mujoco.mj_forward(inner.model, inner.data)
            for _ in range(3):
                for camera, actual in _images(env).items():
                    np.testing.assert_array_equal(actual, expected[camera], err_msg=camera)


def test_explicit_quality_overrides_reach_framebuffer_and_model(env_factory):
    env = env_factory(
        config=_config(RenderingQuality(msaa_samples=0, shadow_map_size=1024, shadows=False)),
        render_mode="rgb_array",
    )
    env.reset(seed=17)
    env.render()
    assert env.unwrapped.model.vis.quality.shadowsize == 1024
    for metadata in env.unwrapped.rendering_metadata["renderers"].values():
        assert metadata["actual_msaa_samples"] == 0
        assert metadata["shadow_map_size"] == 1024
        assert not metadata["shadows"]


def test_image_sensitive_policy_episode_repeats(env_factory, tmp_path):
    from so101_nexus.policy_adapters import RolloutRecorder

    class ImagePolicy:
        def reset(self):
            pass

        def select_action(self, batch):
            means = np.array(
                [batch[f"observation.images.{name}"].mean() for name in ("wrist", "overhead")]
            )
            return batch["observation.state"] + np.tile(means / 255, 3)

    class Dataset:
        def __init__(self, root):
            self.root = root
            self.num_episodes = 0
            self.frames = []

        def add_frame(self, frame):
            self.frames.append(frame)

        def save_episode(self):
            self.num_episodes += 1

    results, rewards = [], []
    for episode in range(2):
        config = _config(RenderingQuality(profile="reproducible"))
        config.obs_mode = "visual"
        env = env_factory(config=config, render_mode="rgb_array", max_episode_steps=6)
        dataset = Dataset(tmp_path / str(episode))
        recorder = RolloutRecorder(env, ImagePolicy(), dataset=dataset, record_side_video=True)
        results.append(recorder.record_episode(seed=11))
        rewards.append([frame["reward"] for frame in dataset.frames])
    np.testing.assert_array_equal(results[0].actions_deg, results[1].actions_deg)
    np.testing.assert_array_equal(results[0].states_deg, results[1].states_deg)
    np.testing.assert_array_equal(rewards[0], rewards[1])
    assert results[0].n_steps == results[1].n_steps == 6
    assert results[0].success == results[1].success
