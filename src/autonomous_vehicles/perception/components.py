"""Four-connected segmentation components as a transparent detection baseline.

This converts predicted masks to boxes; it is not a learned object detector.
Objects touching in the mask merge into a single component.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np

from autonomous_vehicles.perception.boxes import Box, Detection


@dataclass(frozen=True)
class Component:
    box: Box
    pixels: int
    category: int
    centroid: tuple[float, float]


def connected_components(
    mask: np.ndarray, category: int = 2, minimum_pixels: int = 4
) -> tuple[Component, ...]:
    mask = np.asarray(mask)
    if mask.ndim != 2 or not np.issubdtype(mask.dtype, np.integer):
        raise ValueError("Expected a two-dimensional integer mask")
    if minimum_pixels < 1 or category < 0:
        raise ValueError("Invalid component parameters")
    height, width = mask.shape
    visited = np.zeros(mask.shape, dtype=bool)
    result = []
    for y in range(height):
        for x in range(width):
            if visited[y, x] or mask[y, x] != category:
                continue
            queue = deque([(y, x)])
            visited[y, x] = True
            coordinates = []
            while queue:
                row, column = queue.popleft()
                coordinates.append((row, column))
                for next_row, next_column in (
                    (row - 1, column),
                    (row + 1, column),
                    (row, column - 1),
                    (row, column + 1),
                ):
                    if (
                        0 <= next_row < height
                        and 0 <= next_column < width
                        and not visited[next_row, next_column]
                        and mask[next_row, next_column] == category
                    ):
                        visited[next_row, next_column] = True
                        queue.append((next_row, next_column))
            if len(coordinates) < minimum_pixels:
                continue
            rows, columns = np.array(coordinates).T
            box = Box(
                float(columns.min()),
                float(rows.min()),
                float(columns.max() + 1),
                float(rows.max() + 1),
            )
            result.append(
                Component(
                    box,
                    len(coordinates),
                    category,
                    (float(columns.mean() + 0.5), float(rows.mean() + 0.5)),
                )
            )
    return tuple(result)


def mask_detections(
    probabilities: np.ndarray,
    category: int = 2,
    threshold: float = 0.5,
    minimum_pixels: int = 4,
) -> tuple[Detection, ...]:
    probabilities = np.asarray(probabilities)
    if probabilities.ndim != 3 or not 0 <= category < probabilities.shape[0]:
        raise ValueError("Expected class probabilities CHW and an existing category")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Probabilities must be finite and in [0, 1]")
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must lie in [0, 1]")
    selected = (probabilities.argmax(0) == category) & (probabilities[category] >= threshold)
    mask = np.where(selected, category + 1, 0)
    components = connected_components(mask, category + 1, minimum_pixels)
    result = []
    for component in components:
        box = component.box
        region = probabilities[category, int(box.y1) : int(box.y2), int(box.x1) : int(box.x2)]
        foreground = selected[int(box.y1) : int(box.y2), int(box.x1) : int(box.x2)]
        result.append(Detection(box, float(region[foreground].mean()), category))
    return tuple(result)
