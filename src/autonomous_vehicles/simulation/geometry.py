"""Planar metric geometry shared by simulation, planning, and controllers.

World coordinates use x forward at heading zero, y left, and angles in radians.
These helpers do not depend on PyTorch or a simulator backend.
"""

from dataclasses import dataclass
from math import atan2, cos, hypot, isfinite, pi, sin

import numpy as np


def finite(*values: float) -> None:
    """Reject NaN and infinities at public numeric boundaries."""
    if not all(isfinite(value) for value in values):
        raise ValueError("Expected finite numeric values")


def wrap_angle(angle: float) -> float:
    finite(angle)
    return (angle + pi) % (2 * pi) - pi


def angle_difference(target: float, source: float) -> float:
    return wrap_angle(target - source)


def clamp(value: float, lower: float, upper: float) -> float:
    finite(value, lower, upper)
    if lower > upper:
        raise ValueError("Lower bound exceeds upper bound")
    return max(lower, min(value, upper))


@dataclass(frozen=True)
class Vec2:
    x: float = 0.0
    y: float = 0.0

    def __post_init__(self):
        finite(self.x, self.y)

    def __add__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, scale: float) -> "Vec2":
        return Vec2(self.x * scale, self.y * scale)

    def dot(self, other: "Vec2") -> float:
        return self.x * other.x + self.y * other.y

    def cross(self, other: "Vec2") -> float:
        return self.x * other.y - self.y * other.x

    @property
    def norm(self) -> float:
        return hypot(self.x, self.y)

    def normalized(self) -> "Vec2":
        if self.norm <= 1e-12:
            raise ValueError("Cannot normalize zero vector")
        return self * (1 / self.norm)

    def rotate(self, angle: float) -> "Vec2":
        finite(angle)
        return Vec2(
            cos(angle) * self.x - sin(angle) * self.y, sin(angle) * self.x + cos(angle) * self.y
        )

    def array(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float64)


@dataclass(frozen=True)
class Pose2:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0

    def __post_init__(self):
        finite(self.x, self.y, self.yaw)

    @property
    def position(self) -> Vec2:
        return Vec2(self.x, self.y)

    def local_to_world(self, point: Vec2) -> Vec2:
        return self.position + point.rotate(self.yaw)

    def world_to_local(self, point: Vec2) -> Vec2:
        return (point - self.position).rotate(-self.yaw)

    def compose(self, other: "Pose2") -> "Pose2":
        point = self.local_to_world(other.position)
        return Pose2(point.x, point.y, wrap_angle(self.yaw + other.yaw))

    def inverse(self) -> "Pose2":
        point = (self.position * -1).rotate(-self.yaw)
        return Pose2(point.x, point.y, wrap_angle(-self.yaw))


@dataclass(frozen=True)
class SegmentProjection:
    point: Vec2
    fraction: float
    distance: float
    signed_distance: float


def project_segment(point: Vec2, start: Vec2, end: Vec2) -> SegmentProjection:
    delta = end - start
    if delta.norm <= 1e-12:
        return SegmentProjection(start, 0.0, (point - start).norm, 0.0)
    fraction = clamp((point - start).dot(delta) / delta.dot(delta), 0, 1)
    closest = start + delta * fraction
    distance = (point - closest).norm
    side = delta.cross(point - start)
    signed = distance if side >= 0 else -distance
    return SegmentProjection(closest, fraction, distance, signed)


def interpolate_pose(start: Pose2, end: Pose2, fraction: float) -> Pose2:
    fraction = clamp(fraction, 0, 1)
    return Pose2(
        start.x + fraction * (end.x - start.x),
        start.y + fraction * (end.y - start.y),
        wrap_angle(start.yaw + fraction * angle_difference(end.yaw, start.yaw)),
    )


def heading(start: Vec2, end: Vec2) -> float:
    delta = end - start
    if delta.norm <= 1e-12:
        raise ValueError("Heading needs distinct points")
    return atan2(delta.y, delta.x)


def polyline_lengths(points: tuple[Vec2, ...]) -> np.ndarray:
    if len(points) < 2:
        raise ValueError("A polyline needs at least two points")
    lengths = np.array([(b - a).norm for a, b in zip(points[:-1], points[1:], strict=True)])
    if np.any(lengths <= 1e-9):
        raise ValueError("Consecutive polyline points must differ")
    return np.concatenate(([0.0], np.cumsum(lengths)))
