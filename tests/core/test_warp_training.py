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
