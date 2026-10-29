"""Local lateral path candidates with explicit feasibility and cost components."""

from dataclasses import dataclass

import numpy as np

from autonomous_vehicles.planning.path import Path, path_from_xy
from autonomous_vehicles.planning.polynomial import Boundary, Quintic
from autonomous_vehicles.simulation.geometry import finite
from autonomous_vehicles.simulation.roads import Road
from autonomous_vehicles.simulation.shapes import Rectangle, point_rectangle_distance


@dataclass(frozen=True)
class LatticeConfig:
    horizon: float = 30.0
    spacing: float = 0.5
    offsets: tuple[float, ...] = (-2.0, -1.0, 0.0, 1.0, 2.0)
    clearance: float = 1.5
    maximum_curvature: float = 0.25
    deviation_weight: float = 1.0
    curvature_weight: float = 10.0

    def __post_init__(self):
        finite(
            self.horizon,
            self.spacing,
            self.clearance,
            self.maximum_curvature,
            self.deviation_weight,
            self.curvature_weight,
            *self.offsets,
        )
        if min(self.horizon, self.spacing, self.maximum_curvature) <= 0 or not self.offsets:
            raise ValueError("Invalid lattice geometry")
        if min(self.clearance, self.deviation_weight, self.curvature_weight) < 0:
            raise ValueError("Margins and costs cannot be negative")
        if len(set(self.offsets)) != len(self.offsets):
            raise ValueError("Offsets must be unique")


@dataclass(frozen=True)
class Candidate:
    path: Path
    offset: float
    feasible: bool
    cost: float
    reason: str
    minimum_clearance: float | None


class LatticePlanner:
    def __init__(self, config: LatticeConfig | None = None):
        self.config = LatticeConfig() if config is None else config

    def candidates(
        self,
        road: Road,
        start_s: float,
        start_lateral: float = 0.0,
        obstacles: tuple[Rectangle, ...] = (),
    ) -> tuple[Candidate, ...]:
        finite(start_s, start_lateral)
        if not 0 <= start_s < road.length:
            raise ValueError("Start arc length must precede road end")
        length = min(self.config.horizon, road.length - start_s)
        count = max(3, int(np.ceil(length / self.config.spacing)) + 1)
        distances = np.linspace(0, length, count)
        result = []
        for offset in self.config.offsets:
            transition = Quintic(Boundary(start_lateral), Boundary(offset), length)
            lateral = transition.value(distances)
            poses = [
                road.pose_at(start_s + float(s), float(d))
                for s, d in zip(distances, lateral, strict=True)
            ]
            path = path_from_xy(np.array([[pose.x, pose.y] for pose in poses]))
            clearances = [
                point_rectangle_distance(point.pose.position, obstacle)
                for point in path.points
                for obstacle in obstacles
            ]
            minimum = min(clearances) if clearances else None
            peak_curvature = max(abs(point.curvature) for point in path.points)
            reason = "feasible"
            if np.max(np.abs(lateral)) + self.config.clearance > road.width / 2:
                reason = "road_boundary"
            elif minimum is not None and minimum < self.config.clearance:
                reason = "obstacle_clearance"
            elif peak_curvature > self.config.maximum_curvature:
                reason = "curvature"
            cost = self.config.deviation_weight * offset**2 + self.config.curvature_weight * sum(
                point.curvature**2 for point in path.points
            )
            result.append(Candidate(path, offset, reason == "feasible", cost, reason, minimum))
        return tuple(result)

    def select(self, candidates: tuple[Candidate, ...]) -> Candidate | None:
        feasible = [candidate for candidate in candidates if candidate.feasible]
        return (
            min(feasible, key=lambda candidate: (candidate.cost, abs(candidate.offset)))
            if feasible
            else None
        )
