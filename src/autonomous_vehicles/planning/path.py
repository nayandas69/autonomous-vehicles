"""Immutable sampled paths and time-parameterized trajectories."""

from dataclasses import dataclass
from math import atan2

import numpy as np

from autonomous_vehicles.simulation.geometry import Pose2, Vec2, finite, interpolate_pose


@dataclass(frozen=True)
class PathPoint:
    pose: Pose2
    s: float
    curvature: float = 0.0

    def __post_init__(self):
        finite(self.s, self.curvature)
        if self.s < 0:
            raise ValueError("Arc length must be nonnegative")


@dataclass(frozen=True)
class Path:
    points: tuple[PathPoint, ...]

    def __post_init__(self):
        if len(self.points) < 2:
            raise ValueError("Path needs at least two points")
        if self.points[0].s != 0:
            raise ValueError("Path must begin at arc length zero")
        if any(b.s <= a.s for a, b in zip(self.points[:-1], self.points[1:], strict=True)):
            raise ValueError("Path arc length must strictly increase")

    @property
    def length(self) -> float:
        return self.points[-1].s

    def pose_at(self, s: float) -> Pose2:
        finite(s)
        if s <= 0:
            return self.points[0].pose
        if s >= self.length:
            return self.points[-1].pose
        coordinates = [point.s for point in self.points]
        index = int(np.searchsorted(coordinates, s))
        a, b = self.points[index - 1 : index + 1]
        return interpolate_pose(a.pose, b.pose, (s - a.s) / (b.s - a.s))

    def positions(self) -> np.ndarray:
        return np.array([[point.pose.x, point.pose.y] for point in self.points])

    def nearest(self, position: Vec2) -> PathPoint:
        return min(self.points, key=lambda point: (point.pose.position - position).norm)


@dataclass(frozen=True)
class TrajectoryPoint:
    time: float
    pose: Pose2
    speed: float
    acceleration: float = 0.0

    def __post_init__(self):
        finite(self.time, self.speed, self.acceleration)
        if self.time < 0 or self.speed < 0:
            raise ValueError("Time and speed must be nonnegative")


@dataclass(frozen=True)
class Trajectory:
    points: tuple[TrajectoryPoint, ...]

    def __post_init__(self):
        if len(self.points) < 2:
            raise ValueError("Trajectory needs at least two points")
        if any(b.time <= a.time for a, b in zip(self.points[:-1], self.points[1:], strict=True)):
            raise ValueError("Trajectory times must strictly increase")

    @property
    def duration(self) -> float:
        return self.points[-1].time - self.points[0].time

    def sample(self, time: float) -> TrajectoryPoint:
        finite(time)
        if time <= self.points[0].time:
            return self.points[0]
        if time >= self.points[-1].time:
            return self.points[-1]
        index = int(np.searchsorted([point.time for point in self.points], time))
        a, b = self.points[index - 1 : index + 1]
        fraction = (time - a.time) / (b.time - a.time)
        return TrajectoryPoint(
            time,
            interpolate_pose(a.pose, b.pose, fraction),
            a.speed + fraction * (b.speed - a.speed),
            a.acceleration + fraction * (b.acceleration - a.acceleration),
        )


def path_from_xy(coordinates: np.ndarray) -> Path:
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2 or len(coordinates) < 2:
        raise ValueError("Expected at least two XY points")
    if not np.isfinite(coordinates).all():
        raise ValueError("Path coordinates must be finite")
    delta = np.diff(coordinates, axis=0)
    distances = np.linalg.norm(delta, axis=1)
    if np.any(distances < 1e-9):
        raise ValueError("Adjacent points must differ")
    arc = np.concatenate(([0], np.cumsum(distances)))
    headings = [atan2(y, x) for x, y in delta]
    headings.append(headings[-1])
    curvature = np.gradient(np.unwrap(headings), arc)
    return Path(
        tuple(
            PathPoint(Pose2(float(x), float(y), float(yaw)), float(s), float(k))
            for (x, y), yaw, s, k in zip(coordinates, headings, arc, curvature, strict=True)
        )
    )
