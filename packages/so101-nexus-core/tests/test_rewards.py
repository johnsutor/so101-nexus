import pytest

from so101_nexus_core.rewards import orientation_progress, reach_progress, simple_reward


class TestReachProgress:
    def test_zero_distance_returns_one(self):
        assert reach_progress(0.0, scale=5.0) == pytest.approx(1.0)

    def test_large_distance_returns_near_zero(self):
        assert reach_progress(10.0, scale=5.0) == pytest.approx(0.0, abs=0.01)

    def test_monotonically_decreasing(self):
        vals = [reach_progress(d, scale=5.0) for d in [0.0, 0.1, 0.5, 1.0, 5.0]]
        for i in range(len(vals) - 1):
            assert vals[i] > vals[i + 1]

    def test_returns_float(self):
        assert isinstance(reach_progress(0.5, scale=5.0), float)


class TestOrientationProgress:
    def test_same_direction_returns_one(self):
        assert orientation_progress(1.0) == pytest.approx(1.0)

    def test_opposite_direction_returns_zero(self):
        assert orientation_progress(-1.0) == pytest.approx(0.0)

    def test_perpendicular_returns_half(self):
        assert orientation_progress(0.0) == pytest.approx(0.5)

    def test_clamps_above_one(self):
        assert orientation_progress(1.1) == pytest.approx(1.0)

    def test_clamps_below_neg_one(self):
        assert orientation_progress(-1.1) == pytest.approx(0.0)


class TestSimpleReward:
    """Tests for the reach/orient/move reward: (1-bonus)*progress + bonus*success."""

    def test_no_success_no_bonus(self):
        r = simple_reward(progress=0.5, completion_bonus=0.1, success=False)
        assert r == pytest.approx(0.9 * 0.5)

    def test_success_adds_bonus(self):
        r = simple_reward(progress=0.5, completion_bonus=0.1, success=True)
        assert r == pytest.approx(0.9 * 0.5 + 0.1)

    def test_zero_progress_no_success(self):
        r = simple_reward(progress=0.0, completion_bonus=0.1, success=False)
        assert r == pytest.approx(0.0)

    def test_full_progress_with_success(self):
        r = simple_reward(progress=1.0, completion_bonus=0.1, success=True)
        assert r == pytest.approx(1.0)
