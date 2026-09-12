"""PickReturn configuration and adversarial reward trajectories."""

import numpy as np
import pytest

from so101_nexus import PickReturnConfig, RestingJointPositions
from so101_nexus.observations import observations_from_feature_names, privileged_state_feature_names
from so101_nexus.rewards import pick_return_success, pick_return_task_potential, potential_shaping


def potential(height=0.0, error=90.0, speed=1.0, grasped=True):
    return pick_return_task_potential(
        height,
        error,
        speed,
        grasped,
        lift_threshold=0.05,
        return_threshold_deg=5.0,
        scale=5.0,
        velocity_scale=15.0,
    )


def test_defaults_and_recorded_rest_target_schema():
    config = PickReturnConfig()
    names = privileged_state_feature_names(config.observations)
    assert all(f"resting_joint_positions_{i}" in names for i in range(5))
    assert privileged_state_feature_names(observations_from_feature_names(names)) == names
    assert RestingJointPositions().size == 5
    assert config.return_threshold_deg == 5.0
    assert config.reward.completion_bonus > max(
        config.reward.reaching, config.reward.grasping, config.reward.task_objective
    )
    assert (
        config.task_description
        == "Pick up the selected object and return the arm to rest while holding it."
    )


@pytest.mark.parametrize("field", ["return_threshold_deg", "lift_threshold", "max_goal_height"])
@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_thresholds(field, value):
    with pytest.raises(ValueError, match=field):
        PickReturnConfig(**{field: value})


def test_empty_and_custom_observation_lists_are_preserved():
    for observations in ([], [RestingJointPositions()]):
        assert PickReturnConfig(observations=observations).observations == observations


def test_ideal_lift_return_settle_is_monotone():
    trajectory = [
        potential(grasped=False),
        potential(),
        potential(height=0.025),
        potential(height=0.06),
        potential(height=0.06, error=30),
        potential(height=0.06, error=3),
        potential(height=0.06, error=3, speed=0),
    ]
    assert np.all(np.diff(trajectory) >= 0)
    assert trajectory[-1] == pytest.approx(1.0)
    assert potential(height=0.5, error=90) == potential(height=0.06, error=90)


def test_return_shaping_has_useful_credit_far_from_rest():
    assert potential(height=0.06, error=60) - potential(height=0.06, error=90) > 0.01


def test_empty_rest_dwelling_and_drop_regrasp_cannot_pay():
    assert potential(height=0.06, error=0, speed=0, grasped=False) == 0
    held = potential(height=0.06, error=30)
    assert potential_shaping(held, held) == 0
    assert potential_shaping(0.0, held) < 0
    assert potential_shaping(0.0, held) + potential_shaping(held, 0.0) == 0
    assert potential(height=0.06, error=30, speed=0) == held


def test_grasp_holds_reach_credit_for_different_object_offsets():
    from so101_nexus.rewards import pick_return_reach_potential

    for distance in (0.0, 0.02, 0.2):
        assert pick_return_reach_potential(distance, True, scale=5.0) == 1.0


def test_static_shaping_scale_changes_final_settle(recwarn):
    from so101_nexus import RewardConfig

    config = PickReturnConfig(reward=RewardConfig(velocity_shaping_scale=2.0))
    assert not recwarn
    slow = pick_return_task_potential(
        0.06,
        0,
        0.3,
        True,
        lift_threshold=config.lift_threshold,
        return_threshold_deg=config.return_threshold_deg,
        scale=5,
        velocity_scale=config.reward.velocity_shaping_scale,
    )
    assert slow > potential(height=0.06, error=0, speed=0.3)


def test_max_goal_height_shapes_lift_without_rewarding_excess_height():
    def shaped(height, cap):
        return pick_return_task_potential(
            height,
            90,
            1,
            True,
            lift_threshold=0.05,
            return_threshold_deg=5,
            scale=5,
            velocity_scale=15,
            max_goal_height=cap,
        )

    assert shaped(0.025, 0.04) > shaped(0.025, 0.16)
    assert shaped(0.06, 0.04) == pytest.approx(shaped(0.5, 0.16))


@pytest.mark.parametrize("library", ["numpy", "torch"])
def test_batched_potential_and_success(library):
    def array(values):
        if library == "torch":
            return pytest.importorskip("torch").tensor(values)
        return np.asarray(values)

    height, error, speed = array([0.06, 0.06, 0.0]), array([0.0, 30.0, 0.0]), array([0.0] * 3)
    grasped = array([True, True, True])
    actual = potential(height, error, speed, grasped)
    np.testing.assert_allclose(actual, [1, potential(0.06, 30, 0), 0], atol=1e-6)
    success = pick_return_success(
        height,
        error,
        grasped,
        array([True] * 3),
        lift_threshold=0.05,
        return_threshold_deg=5.0,
    )
    assert success.tolist() == [True, False, False]
