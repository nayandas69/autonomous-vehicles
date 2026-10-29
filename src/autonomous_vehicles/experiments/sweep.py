"""Small deterministic hyperparameter sweeps ranked using validation only."""

from dataclasses import dataclass
from itertools import product

import numpy as np
import torch
from torch.utils.data import DataLoader

from autonomous_vehicles.data import RoadScenes
from autonomous_vehicles.experiments.reproducibility import reproducible
from autonomous_vehicles.experiments.training import fit_epochs
from autonomous_vehicles.model import RoadNet


@dataclass(frozen=True)
class SweepTrial:
    seed: int
    learning_rate: float
    loss: str
    validation_iou: float
    final_training_loss: float


def perception_sweep(
    seeds: tuple[int, ...] = (42, 43),
    learning_rates: tuple[float, ...] = (0.001, 0.003),
    losses: tuple[str, ...] = ("cross_entropy",),
    epochs: int = 2,
    samples: int = 32,
    size: int = 32,
) -> tuple[SweepTrial, ...]:
    if not seeds or not learning_rates or not losses:
        raise ValueError("Sweep dimensions cannot be empty")
    trials = []
    for seed, learning_rate, loss in product(seeds, learning_rates, losses):
        with reproducible(seed):
            model = RoadNet()
            train = DataLoader(
                RoadScenes(samples, size, seed),
                batch_size=8,
                shuffle=True,
                generator=torch.Generator().manual_seed(seed),
            )
            validation = DataLoader(RoadScenes(16, size, seed, "validation"), batch_size=8)
            history = fit_epochs(model, train, validation, epochs, loss, learning_rate)
            trials.append(
                SweepTrial(
                    seed,
                    learning_rate,
                    loss,
                    max(row["mean_iou"] for row in history),
                    history[-1]["train_loss"],
                )
            )
    return tuple(trials)


def rank_settings(trials: tuple[SweepTrial, ...]) -> list[dict]:
    groups = {}
    for trial in trials:
        key = trial.learning_rate, trial.loss
        groups.setdefault(key, []).append(trial.validation_iou)
    ranking = [
        {
            "learning_rate": learning_rate,
            "loss": loss,
            "seeds": len(values),
            "validation_iou_mean": float(np.mean(values)),
            "validation_iou_std": float(np.std(values)),
        }
        for (learning_rate, loss), values in groups.items()
    ]
    return sorted(ranking, key=lambda row: -row["validation_iou_mean"])
