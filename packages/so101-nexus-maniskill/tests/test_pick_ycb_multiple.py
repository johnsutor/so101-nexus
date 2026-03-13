import gymnasium as gym
import pytest
import torch
from mani_skill import ASSET_DIR

import so101_nexus_maniskill  # noqa: F401
from so101_nexus_core.config import YCB_OBJECTS, PickYCBMultipleConfig
from so101_nexus_maniskill.pick_ycb_multiple import PICK_YCB_MULTIPLE_CONFIGS

_CFG = PickYCBMultipleConfig()
BASE_KWARGS = dict(obs_mode="state", num_envs=1, render_mode=None)


def _has_ycb_assets() -> bool:
    manifest = ASSET_DIR / "assets" / "mani_skill2_ycb" / "info_pick_v0.json"
    return manifest.exists()


def _skip_if_missing_ycb_assets() -> None:
    if _has_ycb_assets():
        return
    pytest.skip(
        "Missing ManiSkill YCB assets at "
        f"{ASSET_DIR / 'assets' / 'mani_skill2_ycb' / 'info_pick_v0.json'}"
    )


def _make_ycb_env(env_id: str, **kwargs):
    _skip_if_missing_ycb_assets()
    return gym.make(env_id, **kwargs)


LIFT_ENV_IDS = [
    ("ManiSkillPickYCBMultipleLiftSO100-v1", "so100"),
    ("ManiSkillPickYCBMultipleLiftSO101-v1", "so101"),
]
ALL_ENV_IDS = LIFT_ENV_IDS


@pytest.fixture(scope="module")
def lift_so100_env():
    env = _make_ycb_env("ManiSkillPickYCBMultipleLiftSO100-v1", **BASE_KWARGS)
    yield env
    env.close()


@pytest.fixture(scope="module")
def lift_so101_env():
    env = _make_ycb_env("ManiSkillPickYCBMultipleLiftSO101-v1", **BASE_KWARGS)
    yield env
    env.close()


def _get_env(request, env_id):
    mapping = {
        "ManiSkillPickYCBMultipleLiftSO100-v1": "lift_so100_env",
        "ManiSkillPickYCBMultipleLiftSO101-v1": "lift_so101_env",
    }
    return request.getfixturevalue(mapping[env_id])


class TestConstructionValidation:
    def test_invalid_available_model_ids(self):
        with pytest.raises(ValueError, match="available_model_ids"):
            PickYCBMultipleConfig(available_model_ids=("invalid_object",))

    def test_invalid_num_distractors(self):
        with pytest.raises(ValueError, match="num_distractors"):
            PickYCBMultipleConfig(num_distractors=0)


class TestEnvCreation:
    @pytest.mark.parametrize("env_id,robot", ALL_ENV_IDS)
    def test_env_creates(self, request, env_id, robot):
        env = _get_env(request, env_id)
        assert isinstance(env, gym.Env)

    @pytest.mark.parametrize("env_id,robot", ALL_ENV_IDS)
    def test_env_reset(self, request, env_id, robot):
        env = _get_env(request, env_id)
        obs, info = env.reset()
        assert isinstance(obs, torch.Tensor)
        assert isinstance(info, dict)

    @pytest.mark.parametrize("env_id,robot", ALL_ENV_IDS)
    def test_env_step(self, request, env_id, robot):
        env = _get_env(request, env_id)
        env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(obs, torch.Tensor)
        assert reward is not None

    @pytest.mark.parametrize("env_id,robot", ALL_ENV_IDS)
    def test_observation_space(self, request, env_id, robot):
        env = _get_env(request, env_id)
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)

    @pytest.mark.parametrize("env_id,robot", ALL_ENV_IDS)
    def test_action_space_shape(self, request, env_id, robot):
        env = _get_env(request, env_id)
        assert env.action_space.shape == (6,)


class TestEpisodeLogic:
    EVALUATE_KEYS = {
        "is_grasped",
        "lift_height",
        "success",
        "tcp_to_obj_dist",
    }

    @pytest.mark.parametrize("env_id,robot", ALL_ENV_IDS)
    def test_evaluate_keys(self, request, env_id, robot):
        env = _get_env(request, env_id)
        env.reset()
        info = env.unwrapped.evaluate()
        assert set(info.keys()) == self.EVALUATE_KEYS

    @pytest.mark.parametrize("env_id,robot", ALL_ENV_IDS)
    def test_target_spawns_in_radius_bounds(self, request, env_id, robot):
        env = _get_env(request, env_id)
        env.reset()
        cfg = PICK_YCB_MULTIPLE_CONFIGS[robot]
        min_r = cfg["spawn_min_radius"]
        max_r = cfg["spawn_max_radius"]
        obj_p = env.unwrapped.obj.pose.p[0].cpu()
        r = float(torch.sqrt(obj_p[0] ** 2 + obj_p[1] ** 2))
        assert min_r <= r <= max_r

    @pytest.mark.parametrize("env_id,robot", LIFT_ENV_IDS)
    def test_reward_range_lift(self, request, env_id, robot):
        env = _get_env(request, env_id)
        env.reset()
        action = env.action_space.sample()
        _, reward, _, _, _ = env.step(action)
        assert (reward >= 0).all()
        assert (reward <= 1).all()


class TestMultipleObjects:
    def test_correct_number_of_distractors(self, lift_so100_env):
        inner = lift_so100_env.unwrapped
        assert len(inner.distractors) == _CFG.num_distractors

    def test_distractor_models_differ_from_target(self, lift_so100_env):
        inner = lift_so100_env.unwrapped
        for mid in inner.distractor_model_ids:
            assert mid != inner.model_id
            assert mid in YCB_OBJECTS


class TestTaskDescription:
    def test_task_description_starts_with_capital(self, lift_so100_env):
        assert lift_so100_env.unwrapped.task_description[0].isupper()

    def test_task_description_includes_object_name(self, lift_so100_env):
        assert "golf ball" in lift_so100_env.unwrapped.task_description
