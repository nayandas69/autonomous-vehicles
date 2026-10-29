"""Pixel calibration summaries and validation-only temperature fitting."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from autonomous_vehicles.perception.losses import validate_logits


@dataclass(frozen=True)
class CalibrationBin:
    lower: float
    upper: float
    count: int
    confidence: float | None
    accuracy: float | None


@dataclass(frozen=True)
class CalibrationReport:
    expected_error: float
    brier_score: float
    negative_log_likelihood: float
    bins: tuple[CalibrationBin, ...]


@torch.inference_mode()
def calibration_report(
    logits: torch.Tensor, targets: torch.Tensor, bins: int = 10
) -> CalibrationReport:
    validate_logits(logits, targets)
    if bins < 2:
        raise ValueError("Need at least two calibration bins")
    probabilities = logits.softmax(1)
    confidence, prediction = probabilities.max(1)
    correct = prediction.eq(targets)
    result, expected = [], 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        selected = (confidence >= lower) & (
            (confidence <= upper) if index == bins - 1 else (confidence < upper)
        )
        count = int(selected.sum())
        average_confidence = confidence[selected].mean().item() if count else None
        accuracy = correct[selected].float().mean().item() if count else None
        if count:
            expected += count / targets.numel() * abs(average_confidence - accuracy)
        result.append(CalibrationBin(lower, upper, count, average_confidence, accuracy))
    one_hot = F.one_hot(targets, logits.shape[1]).permute(0, 3, 1, 2)
    brier = ((probabilities - one_hot) ** 2).sum(1).mean().item()
    return CalibrationReport(
        expected, brier, F.cross_entropy(logits, targets).item(), tuple(result)
    )


def fit_temperature(logits: torch.Tensor, targets: torch.Tensor, steps: int = 100) -> float:
    """Fit one positive scalar on validation data; never fit on test labels."""
    validate_logits(logits, targets)
    if steps < 1:
        raise ValueError("Temperature fitting requires at least one step")
    detached = logits.detach()
    log_temperature = torch.zeros((), device=logits.device, requires_grad=True)
    optimizer = torch.optim.Adam([log_temperature], lr=0.05)
    best_loss = F.cross_entropy(detached, targets).item()
    best = 1.0
    for _ in range(steps):
        optimizer.zero_grad()
        temperature = log_temperature.clamp(-4, 4).exp()
        loss = F.cross_entropy(detached / temperature, targets)
        if loss.item() < best_loss:
            best_loss, best = loss.item(), temperature.item()
        loss.backward()
        optimizer.step()
    return best


@torch.inference_mode()
def predictive_entropy(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 4 or logits.shape[1] < 2 or not torch.isfinite(logits).all():
        raise ValueError("Expected finite BCHW logits with at least two classes")
    log_probabilities = logits.log_softmax(1)
    return -(log_probabilities.exp() * log_probabilities).sum(1)
