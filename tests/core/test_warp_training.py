import pytest
import torch

from examples._warp_training import generalized_advantage_estimate, ppo_losses


@pytest.mark.parametrize("final_done", [0.0, 1.0])
def test_advantages_respect_episode_boundaries_and_bootstrap(final_done):
    rewards = torch.tensor([[1.0], [2.0], [3.0]])
    values = torch.tensor([[0.5], [0.7], [0.9]])
    dones = torch.tensor([[0.0], [1.0], [0.0]])
    advantages, returns = generalized_advantage_estimate(
        rewards, values, dones, torch.tensor([1.1]), torch.tensor([final_done]), 0.9, 0.8
    )
    last = 3.0 + 0.9 * 1.1 * (1.0 - final_done) - 0.9
    expected = torch.tensor([[0.5], [2.0 + 0.9 * 0.9 - 0.7 + 0.72 * last], [last]])
    torch.testing.assert_close(advantages, expected)
    torch.testing.assert_close(returns, expected + values)


@pytest.mark.parametrize("clip_value", [False, True])
@pytest.mark.parametrize("normalize_advantages", [False, True])
def test_ppo_losses_preserve_values_and_gradients(clip_value, normalize_advantages):
    new_log_prob = torch.tensor([0.4, -0.1, -0.3, 0.1], requires_grad=True)
    new_value = torch.tensor([2.0, -1.0, 0.1, 0.9], requires_grad=True)
    advantages = torch.tensor([1.0, 1.0, -1.0, -1.0])
    if normalize_advantages:
        advantages = advantages / advantages.std()
    ratio = new_log_prob.exp()
    expected_policy = torch.maximum(-advantages * ratio, -advantages * ratio.clamp(0.8, 1.2)).mean()
    squared_error = (new_value - torch.tensor([0.0, 1.0, 0.0, 1.0])) ** 2
    if clip_value:
        clipped = torch.tensor([0.5, 0.5, 1.0, 0.0]) + (
            new_value - torch.tensor([0.5, 0.5, 1.0, 0.0])
        ).clamp(-0.2, 0.2)
        squared_error = torch.maximum(
            squared_error, (clipped - torch.tensor([0.0, 1.0, 0.0, 1.0])) ** 2
        )
    expected_value = 0.5 * squared_error.mean()
    expected_gradients = torch.autograd.grad(
        expected_policy + expected_value, (new_log_prob, new_value), retain_graph=True
    )
    policy, value, kl, clip_fraction = ppo_losses(
        new_log_prob,
        torch.zeros(4),
        new_value,
        torch.tensor([0.5, 0.5, 1.0, 0.0]),
        torch.tensor([1.0, 1.0, -1.0, -1.0]),
        torch.tensor([0.0, 1.0, 0.0, 1.0]),
        0.2,
        normalize_advantages,
        clip_value,
    )
    torch.testing.assert_close(policy, expected_policy)
    torch.testing.assert_close(value, expected_value)
    torch.testing.assert_close(kl, ((ratio - 1) - new_log_prob).mean())
    assert clip_fraction.item() == 0.5
    assert not kl.requires_grad
    assert not clip_fraction.requires_grad
    for actual, expected in zip(
        torch.autograd.grad(policy + value, (new_log_prob, new_value)),
        expected_gradients,
        strict=True,
    ):
        torch.testing.assert_close(actual, expected)


def test_single_advantage_is_not_normalized():
    policy, _, _, _ = ppo_losses(
        torch.zeros(1, requires_grad=True),
        torch.zeros(1),
        torch.zeros(1),
        torch.zeros(1),
        torch.ones(1),
        torch.zeros(1),
        0.2,
        True,
        True,
    )
    assert policy.item() == -1.0


def test_rollout_buffers_keep_episode_boundaries_and_one_time_success_bonus():
    from types import SimpleNamespace

    from examples._warp_training import ObsNormalizer, RewardScaler, RolloutCollector

    class Agent:
        def get_action_and_value(self, observation, *, generator):
            assert not torch.is_grad_enabled()
            action = torch.rand((2, 1), generator=generator)
            return action, observation[:, 0] + 0.1, None, observation[:, 0] + 0.2

    rewards = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]]).float()
    successes = torch.tensor([[True, False], [True, True], [True, True], [False, False]])
    terminated = torch.tensor([[False, False], [True, False], [False, False], [False, False]])
    truncated = torch.tensor([[False, False], [False, False], [False, True], [False, False]])
    position = 0
    actions = []

    def step(action):
        nonlocal position
        actions.append(action.clone())
        i = position
        position += 1
        observation = torch.tensor([[2 * i + 2], [2 * i + 3]]).float()
        return observation, rewards[i], terminated[i], truncated[i], {"success": successes[i]}

    generator = torch.Generator().manual_seed(47)
    collector = RolloutCollector(
        SimpleNamespace(num_envs=2, single_action_space=SimpleNamespace(shape=(1,)), step=step),
        Agent(),
        ObsNormalizer(1, "cpu", enabled=False),
        RewardScaler(2, "cpu", 0.99, enabled=False),
        generator,
        2,
        torch.tensor([[0.0], [1.0]]),
    )
    completed, hold_sum = collector.collect(success_bonus=5)
    assert completed == [[4, 2, 1]]
    assert hold_sum.item() == 1.5
    torch.testing.assert_close(
        collector.observations[:, :, 0], torch.tensor([[0, 1], [2, 3]]).float()
    )
    torch.testing.assert_close(collector.log_probs, torch.tensor([[0.1, 1.1], [2.1, 3.1]]))
    torch.testing.assert_close(collector.values, torch.tensor([[0.2, 1.2], [2.2, 3.2]]))
    torch.testing.assert_close(collector.rewards, torch.tensor([[6, 2], [3, 9]]).float())
    torch.testing.assert_close(collector.dones, torch.zeros(2, 2))
    torch.testing.assert_close(collector.next_obs, torch.tensor([[4.0], [5.0]]))
    torch.testing.assert_close(collector.next_done, torch.tensor([1.0, 0.0]))

    completed, hold_sum = collector.collect(success_bonus=5)
    assert completed == [[12, 3, 1]]
    assert hold_sum.item() == 1
    torch.testing.assert_close(
        collector.observations[:, :, 0], torch.tensor([[4, 5], [6, 7]]).float()
    )
    torch.testing.assert_close(collector.rewards, torch.tensor([[10, 6], [7, 8]]).float())
    torch.testing.assert_close(collector.dones, torch.tensor([[1, 0], [0, 1]]).float())
    torch.testing.assert_close(collector.next_obs, torch.tensor([[8.0], [9.0]]))
    torch.testing.assert_close(collector.next_done, torch.zeros(2))
    expected_generator = torch.Generator().manual_seed(47)
    expected_actions = [torch.rand((2, 1), generator=expected_generator) for _ in range(4)]
    torch.testing.assert_close(torch.stack(actions), torch.stack(expected_actions), rtol=0, atol=0)
    torch.testing.assert_close(collector.actions, torch.stack(expected_actions[2:]), rtol=0, atol=0)
    torch.testing.assert_close(
        generator.get_state(), expected_generator.get_state(), rtol=0, atol=0
    )
