"""Metric occupancy grids for range scans and grid-search experiments.

Cells store occupancy log odds. Unknown cells start at zero log odds, and the
inverse sensor model only updates beams that produced a hit. This avoids
mistaking dropout for observed free space.
"""

from dataclasses import dataclass
from math import ceil, floor, log

import numpy as np

from autonomous_vehicles.simulation.geometry import Vec2, finite
from autonomous_vehicles.simulation.sensors import RangeScan
from autonomous_vehicles.simulation.shapes import Rectangle


@dataclass(frozen=True)
class GridSpec:
    width: int = 120
    height: int = 80
    resolution: float = 0.5
    origin: Vec2 = Vec2(-5, -20)

    def __post_init__(self):
        finite(self.resolution)
        if min(self.width, self.height) < 1 or self.resolution <= 0:
            raise ValueError("Grid dimensions and resolution must be positive")

    def cell(self, point: Vec2) -> tuple[int, int] | None:
        column = floor((point.x - self.origin.x) / self.resolution)
        row = floor((point.y - self.origin.y) / self.resolution)
        return (row, column) if 0 <= row < self.height and 0 <= column < self.width else None

    def center(self, row: int, column: int) -> Vec2:
        if not 0 <= row < self.height or not 0 <= column < self.width:
            raise IndexError((row, column))
        return Vec2(
            self.origin.x + (column + 0.5) * self.resolution,
            self.origin.y + (row + 0.5) * self.resolution,
        )

    @property
    def extent(self) -> tuple[float, float, float, float]:
        return (
            self.origin.x,
            self.origin.x + self.width * self.resolution,
            self.origin.y,
            self.origin.y + self.height * self.resolution,
        )


class OccupancyGrid:
    def __init__(self, spec: GridSpec | None = None):
        self.spec = GridSpec() if spec is None else spec
        self.log_odds = np.zeros((self.spec.height, self.spec.width), dtype=np.float64)
        self.observed = np.zeros_like(self.log_odds, dtype=bool)

    @property
    def probabilities(self) -> np.ndarray:
        return 1 / (1 + np.exp(-self.log_odds))

    def reset(self) -> None:
        self.log_odds.fill(0)
        self.observed.fill(False)

    def integrate(
        self, scan: RangeScan, occupied_probability: float = 0.7, free_probability: float = 0.3
    ):
        if not 0.5 < occupied_probability < 1 or not 0 < free_probability < 0.5:
            raise ValueError("Sensor probabilities must straddle 0.5")
        occupied_update = log(occupied_probability / (1 - occupied_probability))
        free_update = log(free_probability / (1 - free_probability))
        endpoints = scan.points()
        origin = scan.pose.position
        for endpoint, hit in zip(endpoints, scan.hits, strict=True):
            if not hit:
                continue
            end = Vec2(float(endpoint[0]), float(endpoint[1]))
            samples = max(2, int(ceil((end - origin).norm / (self.spec.resolution / 2))) + 1)
            cells = []
            for fraction in np.linspace(0, 1, samples):
                cell = self.spec.cell(origin + (end - origin) * float(fraction))
                if cell is not None and (not cells or cells[-1] != cell):
                    cells.append(cell)
            endpoint_cell = self.spec.cell(end)
            for cell in cells:
                self.observed[cell] = True
                self.log_odds[cell] += occupied_update if cell == endpoint_cell else free_update
        np.clip(self.log_odds, -8, 8, out=self.log_odds)

    def rasterize(self, boxes: tuple[Rectangle, ...]) -> np.ndarray:
        occupied = np.zeros_like(self.observed)
        for row in range(self.spec.height):
            for column in range(self.spec.width):
                point = self.spec.center(row, column)
                occupied[row, column] = any(box.contains(point) for box in boxes)
        return occupied


def inflate_obstacles(occupied: np.ndarray, cells: int) -> np.ndarray:
    occupied = np.asarray(occupied, dtype=bool)
    if occupied.ndim != 2 or cells < 0:
        raise ValueError("Need a two-dimensional occupancy mask and nonnegative radius")
    result = occupied.copy()
    height, width = occupied.shape
    for dy in range(-cells, cells + 1):
        for dx in range(-cells, cells + 1):
            if dx * dx + dy * dy > cells * cells:
                continue
            source_y = slice(max(0, -dy), min(height, height - dy))
            source_x = slice(max(0, -dx), min(width, width - dx))
            target_y = slice(max(0, dy), min(height, height + dy))
            target_x = slice(max(0, dx), min(width, width + dx))
            if abs(dx) < width and abs(dy) < height:
                result[target_y, target_x] |= occupied[source_y, source_x]
    return result
