"""Additional segmentation losses for controlled notebook experiments."""

import torch
import torch.nn.functional as F


def validate_logits(logits: torch.Tensor, targets: torch.Tensor) -> None:
    if logits.ndim != 4 or targets.ndim != 3:
        raise ValueError("Expected BCHW logits and BHW targets")
    if logits.shape[0] != targets.shape[0] or logits.shape[2:] != targets.shape[1:]:
        raise ValueError("Logits and targets must align")
    if targets.dtype != torch.int64 or logits.shape[1] < 2:
        raise ValueError("Need int64 labels and at least two classes")
    if not logits.is_floating_point() or not torch.isfinite(logits).all():
        raise ValueError("Logits must be finite floating point")
    if targets.numel() == 0 or targets.min() < 0 or targets.max() >= logits.shape[1]:
        raise ValueError("Labels outside class range or empty target")


def soft_dice_loss(logits: torch.Tensor, targets: torch.Tensor, smoothing: float = 1.0):
    validate_logits(logits, targets)
    if not 0 < smoothing < float("inf"):
        raise ValueError("Smoothing must be finite and positive")
    probabilities = logits.softmax(1)
    one_hot = F.one_hot(targets, logits.shape[1]).permute(0, 3, 1, 2).to(logits.dtype)
    axes = (0, 2, 3)
    intersection = (probabilities * one_hot).sum(axes)
    denominator = probabilities.sum(axes) + one_hot.sum(axes)
    score = (2 * intersection + smoothing) / (denominator + smoothing)
    return 1 - score.mean()


def focal_loss(logits: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0):
    validate_logits(logits, targets)
    if not 0 <= gamma < float("inf"):
        raise ValueError("Gamma must be finite and nonnegative")
    log_probability = logits.log_softmax(1).gather(1, targets.unsqueeze(1)).squeeze(1)
    return (-((1 - log_probability.exp()) ** gamma) * log_probability).mean()


def combined_loss(logits: torch.Tensor, targets: torch.Tensor, dice_weight: float = 0.5):
    if not 0 <= dice_weight <= 1:
        raise ValueError("Dice weight must lie in [0, 1]")
    validate_logits(logits, targets)
    return (1 - dice_weight) * F.cross_entropy(logits, targets) + dice_weight * soft_dice_loss(
        logits, targets
    )


def inverse_frequency_weights(counts: torch.Tensor, maximum_ratio: float = 10.0):
    if counts.ndim != 1 or len(counts) < 2 or not torch.isfinite(counts).all():
        raise ValueError("Expected a finite vector of class counts")
    if torch.any(counts < 0) or counts.sum() <= 0:
        raise ValueError("Class counts must be nonnegative with nonzero total")
    if not 1 <= maximum_ratio < float("inf"):
        raise ValueError("Maximum ratio must be finite and at least one")
    counts = counts.to(torch.float64)
    weights = counts.sum() / counts.clamp_min(1)
    weights = weights.clamp(max=weights.min() * maximum_ratio)
    return (weights / weights.mean()).to(torch.float32)
