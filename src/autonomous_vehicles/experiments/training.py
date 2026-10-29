"""Reusable training primitives for ablations without duplicating the CLI trainer."""

from dataclasses import dataclass
from math import isfinite

import torch
from torch.utils.data import DataLoader

from autonomous_vehicles.metrics import evaluate
from autonomous_vehicles.perception.losses import combined_loss, focal_loss, soft_dice_loss


@dataclass(frozen=True)
class EpochResult:
    loss: float
    samples: int
    batches: int
    gradient_norm_mean: float


@dataclass
class EarlyStopping:
    patience: int = 5
    minimum_improvement: float = 0.0
    best: float | None = None
    bad_epochs: int = 0

    def __post_init__(self):
        if self.patience < 1 or not 0 <= self.minimum_improvement < float("inf"):
            raise ValueError("Invalid early stopping settings")

    def update(self, score: float) -> bool:
        if not isfinite(score):
            raise ValueError("Early stopping score must be finite")
        if self.best is None or score > self.best + self.minimum_improvement:
            self.best, self.bad_epochs = score, 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def choose_loss(name: str):
    choices = {
        "cross_entropy": torch.nn.functional.cross_entropy,
        "dice": soft_dice_loss,
        "focal": focal_loss,
        "combined": combined_loss,
    }
    if name not in choices:
        raise ValueError(f"Unknown loss {name!r}")
    return choices[name]


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_name: str = "cross_entropy",
    gradient_clip: float = 5.0,
) -> EpochResult:
    if not 0 < gradient_clip < float("inf"):
        raise ValueError("Gradient clipping threshold must be finite and positive")
    loss_function = choose_loss(loss_name)
    model.train()
    total_loss, samples, batches, norms = 0.0, 0, 0, []
    for images, targets in loader:
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(model(images), targets)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), gradient_clip, error_if_nonfinite=True
        )
        optimizer.step()
        total_loss += loss.item() * len(images)
        samples += len(images)
        batches += 1
        norms.append(float(norm))
    if samples == 0:
        raise ValueError("Training loader is empty")
    return EpochResult(total_loss / samples, samples, batches, sum(norms) / len(norms))


def fit_epochs(
    model,
    training: DataLoader,
    validation: DataLoader,
    epochs: int = 3,
    loss_name: str = "cross_entropy",
    learning_rate: float = 0.001,
) -> list[dict]:
    if epochs < 1 or not 0 < learning_rate < float("inf"):
        raise ValueError("Invalid fit settings")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    history = []
    for epoch in range(epochs):
        result = train_epoch(model, training, optimizer, loss_name)
        metrics = evaluate(model, validation)
        history.append({"epoch": epoch + 1, "train_loss": result.loss, **metrics})
    return history
