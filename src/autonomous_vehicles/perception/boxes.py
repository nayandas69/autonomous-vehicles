"""Axis-aligned image boxes, overlap measures, and deterministic suppression."""

from dataclasses import dataclass

import numpy as np

from autonomous_vehicles.simulation.geometry import finite


@dataclass(frozen=True)
class Box:
    """Half-open XYXY box in continuous image coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self):
        finite(self.x1, self.y1, self.x2, self.y2)
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("Box must have positive area")

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2

    def intersection(self, other: "Box") -> float:
        return max(0, min(self.x2, other.x2) - max(self.x1, other.x1)) * max(
            0,
            min(self.y2, other.y2) - max(self.y1, other.y1),
        )

    def iou(self, other: "Box") -> float:
        intersection = self.intersection(other)
        return intersection / (self.area + other.area - intersection)

    def clip(self, width: int, height: int) -> "Box | None":
        if width < 1 or height < 1:
            raise ValueError("Image dimensions must be positive")
        x1, x2 = max(0, self.x1), min(width, self.x2)
        y1, y2 = max(0, self.y1), min(height, self.y2)
        return Box(x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None

    def translated(self, dx: float, dy: float) -> "Box":
        return Box(self.x1 + dx, self.y1 + dy, self.x2 + dx, self.y2 + dy)

    def normalized(self, width: int, height: int) -> "Box":
        if width <= 0 or height <= 0:
            raise ValueError("Image dimensions must be positive")
        return Box(self.x1 / width, self.y1 / height, self.x2 / width, self.y2 / height)


@dataclass(frozen=True)
class Detection:
    box: Box
    score: float
    category: int = 2

    def __post_init__(self):
        finite(self.score)
        if not 0 <= self.score <= 1 or self.category < 0:
            raise ValueError("Score must be in [0, 1] and category nonnegative")


def suppress(detections: tuple[Detection, ...], threshold: float = 0.5) -> tuple[Detection, ...]:
    finite(threshold)
    if not 0 <= threshold <= 1:
        raise ValueError("IoU threshold must lie in [0, 1]")
    ordered = sorted(enumerate(detections), key=lambda pair: (-pair[1].score, pair[0]))
    kept = []
    for _, detection in ordered:
        if any(
            existing.category == detection.category and existing.box.iou(detection.box) > threshold
            for existing in kept
        ):
            continue
        kept.append(detection)
    return tuple(kept)


def pairwise_iou(first: tuple[Box, ...], second: tuple[Box, ...]) -> np.ndarray:
    values = np.zeros((len(first), len(second)), dtype=np.float64)
    for row, box in enumerate(first):
        for column, other in enumerate(second):
            values[row, column] = box.iou(other)
    return values
