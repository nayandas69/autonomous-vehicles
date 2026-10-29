"""Streaming segmentation metrics with explicit ignore labels and merge semantics."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SegmentationSummary:
    classes: int
    evaluated_pixels: int
    ignored_pixels: int
    pixel_accuracy: float | None
    mean_iou: float | None
    class_iou: tuple[float | None, ...]
    class_precision: tuple[float | None, ...]
    class_recall: tuple[float | None, ...]


class ConfusionMeter:
    def __init__(self, classes: int = 3, ignore_index: int | None = None):
        if classes < 2 or (ignore_index is not None and 0 <= ignore_index < classes):
            raise ValueError("Need at least two classes and an ignore label outside class IDs")
        self.classes = classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        self.matrix = torch.zeros(self.classes, self.classes, dtype=torch.int64)
        self.ignored_pixels = 0

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        if (
            prediction.shape != target.shape
            or prediction.dtype != torch.int64
            or target.dtype != torch.int64
        ):
            raise ValueError("Prediction and target must be aligned int64 label tensors")
        prediction, target = prediction.detach().cpu(), target.detach().cpu()
        valid = torch.ones_like(target, dtype=torch.bool)
        if self.ignore_index is not None:
            valid = target != self.ignore_index
        selected_target, selected_prediction = target[valid], prediction[valid]
        if selected_target.numel():
            if (
                selected_target.min() < 0
                or selected_target.max() >= self.classes
                or selected_prediction.min() < 0
                or selected_prediction.max() >= self.classes
            ):
                raise ValueError("Labels outside declared class range")
        counts = torch.bincount(
            selected_target * self.classes + selected_prediction, minlength=self.classes**2
        )
        self.matrix += counts.reshape(self.classes, self.classes)
        self.ignored_pixels += int((~valid).sum())

    def merge(self, other: "ConfusionMeter") -> None:
        if self.classes != other.classes or self.ignore_index != other.ignore_index:
            raise ValueError("Cannot merge meters with different contracts")
        self.matrix += other.matrix
        self.ignored_pixels += other.ignored_pixels

    @staticmethod
    def ratios(numerator: torch.Tensor, denominator: torch.Tensor) -> tuple[float | None, ...]:
        return tuple(
            float(n / d) if d > 0 else None for n, d in zip(numerator, denominator, strict=True)
        )

    def summary(self) -> SegmentationSummary:
        matrix = self.matrix.to(torch.float64)
        correct = matrix.diag()
        predicted, actual = matrix.sum(0), matrix.sum(1)
        union = predicted + actual - correct
        class_iou = self.ratios(correct, union)
        present = [value for value in class_iou if value is not None]
        total = int(matrix.sum())
        return SegmentationSummary(
            self.classes,
            total,
            self.ignored_pixels,
            float(correct.sum() / total) if total else None,
            sum(present) / len(present) if present else None,
            class_iou,
            self.ratios(correct, predicted),
            self.ratios(correct, actual),
        )
