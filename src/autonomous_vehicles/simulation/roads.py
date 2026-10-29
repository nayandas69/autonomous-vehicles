"""Arc-length polyline roads with signed lateral projection."""

from dataclasses import dataclass
from math import cos, pi, sin

import numpy as np

from autonomous_vehicles.simulation.geometry import (
    Pose2,
    Vec2,
    clamp,
    finite,
    heading,
    polyline_lengths,
    project_segment,
)


@dataclass(frozen=True)
class RoadProjection:
    s: float
    lateral: float
    distance: float
    pose: Pose2
    segment: int


class Road:
    def __init__(self, points: tuple[Vec2, ...], width: float = 7.0, speed_limit: float = 12.0):
        finite(width, speed_limit)
        if width <= 0 or speed_limit <= 0:
            raise ValueError("Road width and speed limit must be positive")
        self.points = tuple(points)
        self.width = width
        self.speed_limit = speed_limit
        self.lengths = polyline_lengths(self.points)
        self.lengths.setflags(write=False)

    @property
    def length(self) -> float:
        return float(self.lengths[-1])

    def pose_at(self, s: float, lateral: float = 0.0) -> Pose2:
        finite(s, lateral)
        s = clamp(s, 0, self.length)
        index = min(int(np.searchsorted(self.lengths, s, side="right")) - 1, len(self.points) - 2)
        start, end = self.points[index : index + 2]
        fraction = (s - self.lengths[index]) / (self.lengths[index + 1] - self.lengths[index])
        point = start + (end - start) * fraction
        yaw = heading(start, end)
        point = point + Vec2(0, lateral).rotate(yaw)
        return Pose2(point.x, point.y, yaw)

    def project(self, point: Vec2) -> RoadProjection:
        candidates = []
        for index, (start, end) in enumerate(zip(self.points[:-1], self.points[1:], strict=True)):
            result = project_segment(point, start, end)
            s = self.lengths[index] + result.fraction * (
                self.lengths[index + 1] - self.lengths[index]
            )
            candidates.append((result.distance, index, float(s), result.signed_distance))
        distance, index, s, lateral = min(candidates)
        return RoadProjection(s, lateral, distance, self.pose_at(s), index)

    def contains(self, point: Vec2, margin: float = 0) -> bool:
        finite(margin)
        if margin < 0 or margin >= self.width / 2:
            raise ValueError("Margin must be nonnegative and smaller than half road width")
        projection = self.project(point)
        # Endpoint distances also restrict the finite road's extent.
        return projection.distance <= self.width / 2 - margin

    def samples(self, spacing: float = 1.0, lateral: float = 0.0) -> tuple[Pose2, ...]:
        finite(spacing)
        if spacing <= 0:
            raise ValueError("Spacing must be positive")
        count = max(2, int(np.ceil(self.length / spacing)) + 1)
        return tuple(self.pose_at(float(s), lateral) for s in np.linspace(0, self.length, count))

    def curvature_at(self, s: float, window: float = 2.0) -> float:
        from autonomous_vehicles.simulation.geometry import angle_difference

        finite(s, window)
        if window <= 0:
            raise ValueError("Curvature window must be positive")
        before, after = max(0, s - window), min(self.length, s + window)
        if after <= before:
            return 0.0
        return angle_difference(self.pose_at(after).yaw, self.pose_at(before).yaw) / (
            after - before
        )


def straight_road(length: float = 150, width: float = 7) -> Road:
    finite(length)
    if length <= 0:
        raise ValueError("Length must be positive")
    return Road((Vec2(0, 0), Vec2(length, 0)), width)


def curved_road(length: float = 150, amplitude: float = 8, wavelength: float = 100) -> Road:
    finite(length, amplitude, wavelength)
    if length <= 1 or wavelength <= 0:
        raise ValueError("Length must exceed one and wavelength must be positive")
    return Road(
        tuple(
            Vec2(float(x), amplitude * sin(2 * pi * float(x) / wavelength))
            for x in np.linspace(0, length, int(length) + 1)
        )
    )


def circular_road(radius: float = 40, angle: float = pi / 2, samples: int = 101) -> Road:
    finite(radius, angle)
    if radius <= 0 or not 0 < angle <= 2 * pi or samples < 3:
        raise ValueError("Invalid circular road dimensions")
    return Road(
        tuple(
            Vec2(radius * sin(float(a)), radius * (1 - cos(float(a))))
            for a in np.linspace(0, angle, samples)
        )
    )
