"""Curvature and acceleration-constrained speed profiles along sampled paths."""

from dataclasses import dataclass
from math import sqrt

import numpy as np

from autonomous_vehicles.planning.path import Path, Trajectory, TrajectoryPoint
from autonomous_vehicles.simulation.geometry import finite


@dataclass(frozen=True)
class SpeedProfileConfig:
    maximum_speed: float = 10.0
    lateral_acceleration: float = 2.0
    acceleration: float = 2.0
    deceleration: float = 3.0

    def __post_init__(self):
        finite(*vars(self).values())
        if min(vars(self).values()) <= 0:
            raise ValueError("Speed profile limits must be positive")


@dataclass(frozen=True)
class SpeedProfile:
    distances: np.ndarray
    speeds: np.ndarray
    times: np.ndarray

    @property
    def duration(self) -> float:
        return float(self.times[-1])


def plan_speed(
    path: Path,
    start_speed: float = 0,
    end_speed: float = 0,
    config: SpeedProfileConfig | None = None,
) -> SpeedProfile:
    config = SpeedProfileConfig() if config is None else config
    finite(start_speed, end_speed)
    if min(start_speed, end_speed) < 0 or max(start_speed, end_speed) > config.maximum_speed:
        raise ValueError("Boundary speeds outside configured range")
    distances = np.array([point.s for point in path.points])
    curvature = np.array([abs(point.curvature) for point in path.points])
    speeds = np.minimum(
        config.maximum_speed, np.sqrt(config.lateral_acceleration / np.maximum(curvature, 1e-9))
    )
    speeds[0] = min(speeds[0], start_speed)
    speeds[-1] = min(speeds[-1], end_speed)
    for index in range(1, len(speeds)):
        ds = distances[index] - distances[index - 1]
        speeds[index] = min(
            speeds[index], sqrt(speeds[index - 1] ** 2 + 2 * config.acceleration * ds)
        )
    for index in range(len(speeds) - 2, -1, -1):
        ds = distances[index + 1] - distances[index]
        speeds[index] = min(
            speeds[index], sqrt(speeds[index + 1] ** 2 + 2 * config.deceleration * ds)
        )
    average = (speeds[:-1] + speeds[1:]) / 2
    if np.any(average <= 1e-9):
        raise ValueError("Adjacent zero speeds make traversal time undefined; densify the path")
    times = np.concatenate(([0], np.cumsum(np.diff(distances) / average)))
    return SpeedProfile(distances, speeds, times)


def trajectory_from_profile(path: Path, profile: SpeedProfile) -> Trajectory:
    if len(path.points) != len(profile.speeds):
        raise ValueError("Profile and path lengths differ")
    if not np.allclose([point.s for point in path.points], profile.distances):
        raise ValueError("Profile belongs to a different path")
    acceleration = np.gradient(profile.speeds, profile.times)
    return Trajectory(
        tuple(
            TrajectoryPoint(float(time), point.pose, float(speed), float(a))
            for point, time, speed, a in zip(
                path.points, profile.times, profile.speeds, acceleration, strict=True
            )
        )
    )
