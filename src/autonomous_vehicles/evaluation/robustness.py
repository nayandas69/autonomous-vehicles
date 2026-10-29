"""Controlled corruption evaluation using a fixed synthetic test stream."""

from dataclasses import asdict, dataclass

import torch
from torch.utils.data import DataLoader, Dataset

from autonomous_vehicles.data import RoadScenes
from autonomous_vehicles.metrics import evaluate
from autonomous_vehicles.perception.augmentation import image_corruption


class CorruptedDataset(Dataset):
    def __init__(self, base: Dataset, kind: str, severity: float, seed: int = 42):
        if kind not in {"darkness", "fog", "noise", "occlusion"}:
            raise ValueError("Unknown corruption")
        if not 0 <= severity <= 1 or seed < 0:
            raise ValueError("Invalid corruption severity or seed")
        self.base = base
        self.kind = kind
        self.severity = severity
        self.seed = seed

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        image, mask = self.base[index]
        return image_corruption(image, self.kind, self.severity, self.seed + index), mask


@dataclass(frozen=True)
class RobustnessResult:
    corruption: str
    severity: float
    mean_iou: float
    pixel_accuracy: float
    samples: int


def evaluate_corruptions(
    model: torch.nn.Module,
    samples: int = 32,
    size: int = 64,
    seed: int = 42,
    severities: tuple[float, ...] = (0.0, 0.25, 0.5),
    kinds: tuple[str, ...] = ("darkness", "fog", "noise", "occlusion"),
) -> tuple[RobustnessResult, ...]:
    base = RoadScenes(samples, size, seed, "test")
    result = []
    for kind in kinds:
        for severity in severities:
            dataset = CorruptedDataset(base, kind, severity, seed)
            metrics = evaluate(model, DataLoader(dataset, batch_size=16))
            result.append(
                RobustnessResult(
                    kind, severity, metrics["mean_iou"], metrics["pixel_accuracy"], samples
                )
            )
    return tuple(result)


def robustness_table(results: tuple[RobustnessResult, ...]) -> list[dict]:
    return [asdict(result) for result in results]
