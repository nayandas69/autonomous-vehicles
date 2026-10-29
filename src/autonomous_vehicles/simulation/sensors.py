"""Seeded planar range sensor and actor observations in ego coordinates."""

from dataclasses import dataclass
from math import atan2, cos, pi, sin

import numpy as np

from autonomous_vehicles.simulation.actors import ActorState
from autonomous_vehicles.simulation.geometry import Pose2, Vec2, finite, wrap_angle
from autonomous_vehicles.simulation.shapes import ray_rectangle_distance


@dataclass(frozen=True)
class RangeSensorConfig:
    beams: int = 91
    field_of_view: float = pi
    maximum_range: float = 80.0
    noise_std: float = 0.02
    dropout: float = 0.0

    def __post_init__(self):
        finite(self.field_of_view, self.maximum_range, self.noise_std, self.dropout)
        if self.beams < 2 or not 0 < self.field_of_view <= 2 * pi:
            raise ValueError("Need at least two beams and a field of view in (0, 2pi]")
        if self.maximum_range <= 0 or self.noise_std < 0 or not 0 <= self.dropout <= 1:
            raise ValueError("Invalid sensor range, noise or dropout")


@dataclass(frozen=True)
class RangeScan:
    angles: np.ndarray
    ranges: np.ndarray
    hits: np.ndarray
    pose: Pose2

    def points(self, world: bool = True) -> np.ndarray:
        x = self.ranges * np.cos(self.angles)
        y = self.ranges * np.sin(self.angles)
        points = np.column_stack((x, y))
        if not world:
            return points
        c, s = cos(self.pose.yaw), sin(self.pose.yaw)
        rotation = np.array([[c, -s], [s, c]])
        return points @ rotation.T + np.array([self.pose.x, self.pose.y])


class RangeSensor:
    def __init__(self, config: RangeSensorConfig | None = None, seed: int = 42):
        if seed < 0:
            raise ValueError("Seed must be nonnegative")
        self.config = RangeSensorConfig() if config is None else config
        self.rng = np.random.default_rng(seed)

    def scan(self, pose: Pose2, actors: tuple[ActorState, ...]) -> RangeScan:
        config = self.config
        angles = np.linspace(-config.field_of_view / 2, config.field_of_view / 2, config.beams)
        ranges = np.full(config.beams, config.maximum_range, dtype=np.float64)
        hits = np.zeros(config.beams, dtype=bool)
        for index, angle in enumerate(angles):
            direction = Vec2(cos(pose.yaw + angle), sin(pose.yaw + angle))
            for actor in actors:
                distance = ray_rectangle_distance(pose.position, direction, actor.footprint)
                if distance is not None and distance <= ranges[index]:
                    ranges[index] = distance
                    hits[index] = True
        ranges[hits] += self.rng.normal(0, config.noise_std, int(hits.sum()))
        ranges = np.clip(ranges, 0, config.maximum_range)
        dropped = self.rng.random(config.beams) < config.dropout
        hits[dropped] = False
        ranges[dropped] = config.maximum_range
        return RangeScan(angles, ranges, hits, pose)


@dataclass(frozen=True)
class BearingObservation:
    identifier: str
    distance: float
    bearing: float
    relative_position: Vec2


def visible_actors(
    pose: Pose2,
    actors: tuple[ActorState, ...],
    maximum_range: float = 80,
    field_of_view: float = pi,
) -> tuple[BearingObservation, ...]:
    """Center-point visibility only; no occlusion reasoning."""
    finite(maximum_range, field_of_view)
    if maximum_range <= 0 or not 0 < field_of_view <= 2 * pi:
        raise ValueError("Invalid visibility bounds")
    result = []
    for actor in actors:
        local = pose.world_to_local(actor.pose.position)
        bearing = wrap_angle(atan2(local.y, local.x))
        if local.norm <= maximum_range and abs(bearing) <= field_of_view / 2:
            result.append(BearingObservation(actor.identifier, local.norm, bearing, local))
    return tuple(sorted(result, key=lambda observation: observation.distance))
