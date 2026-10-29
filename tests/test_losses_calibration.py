"""Numerical loss properties, gradients, and temperature calibration."""

from math import log

import pytest
import torch
import torch.nn.functional as F

from autonomous_vehicles.perception.calibration import (
    calibration_report,
    fit_temperature,
    predictive_entropy,
)
from autonomous_vehicles.perception.losses import (
    combined_loss,
    focal_loss,
    inverse_frequency_weights,
    soft_dice_loss,
    validate_logits,
)


def test_uniform_logits_cross_entropy_and_focal_reference():
    logits = torch.zeros(1, 3, 2, 2)
    targets = torch.zeros(1, 2, 2, dtype=torch.int64)
    assert focal_loss(logits, targets, gamma=0).item() == pytest.approx(log(3))
    assert focal_loss(logits, targets, gamma=2).item() == pytest.approx((2 / 3) ** 2 * log(3))


@pytest.mark.parametrize("loss_function", [soft_dice_loss, focal_loss, combined_loss])
def test_losses_have_finite_gradients_and_reward_correct_predictions(loss_function):
    target = torch.tensor([[[0, 1], [2, 0]]])
    logits = torch.zeros(1, 3, 2, 2, requires_grad=True)
    loss = loss_function(logits, target)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    correct = F.one_hot(target, 3).permute(0, 3, 1, 2).float() * 10
    assert loss_function(correct, target) < loss


def test_combined_loss_endpoints_match_components():
    logits = torch.zeros(1, 3, 2, 2)
    target = torch.zeros(1, 2, 2, dtype=torch.int64)
    assert combined_loss(logits, target, 0) == F.cross_entropy(logits, target)
    assert combined_loss(logits, target, 1) == soft_dice_loss(logits, target)


def test_class_weights_are_bounded_and_normalized():
    weights = inverse_frequency_weights(torch.tensor([1000, 100, 0]), maximum_ratio=5)
    assert weights.mean().item() == pytest.approx(1)
    assert weights.max() / weights.min() <= 5 + 1e-6
    assert weights[2] >= weights[1] >= weights[0]


def test_uniform_entropy_and_calibration_bin_accounting():
    logits = torch.zeros(1, 3, 2, 2)
    target = torch.zeros(1, 2, 2, dtype=torch.int64)
    entropy = predictive_entropy(logits)
    torch.testing.assert_close(entropy, torch.full((1, 2, 2), log(3)))
    report = calibration_report(logits, target, bins=5)
    assert sum(item.count for item in report.bins) == 4
    assert report.expected_error == pytest.approx(2 / 3)
    assert report.brier_score == pytest.approx(2 / 3)
    assert report.negative_log_likelihood == pytest.approx(log(3))


def test_temperature_fit_does_not_increase_validation_nll():
    logits = torch.tensor([[[[8.0, 8.0, 8.0, 8.0]], [[0.0, 0.0, 0.0, 0.0]]]])
    target = torch.tensor([[[0, 0, 1, 0]]])
    original = logits.clone()
    temperature = fit_temperature(logits, target, steps=30)
    assert temperature > 0
    assert F.cross_entropy(logits / temperature, target) <= F.cross_entropy(logits, target)
    torch.testing.assert_close(logits, original)


@pytest.mark.parametrize(
    "operation",
    [
        lambda: validate_logits(torch.zeros(1, 3, 2, 2), torch.zeros(1, 2, 2)),
        lambda: validate_logits(torch.zeros(1, 3, 2, 2), torch.full((1, 2, 2), 4)),
        lambda: soft_dice_loss(torch.zeros(1, 3, 2, 2), torch.zeros(1, 2, 2, dtype=torch.int64), 0),
        lambda: inverse_frequency_weights(torch.tensor([0, 0])),
        lambda: predictive_entropy(torch.zeros(3, 2, 2)),
    ],
)
def test_invalid_loss_and_calibration_inputs(operation):
    with pytest.raises(ValueError):
        operation()
