"""Convex oriented rectangles and conservative collision queries."""

from dataclasses import dataclass
from math import cos, sin

from autonomous_vehicles.simulation.geometry import Pose2, Vec2, finite


@dataclass(frozen=True)
class Rectangle:
    pose: Pose2
    length: float
    width: float

    def __post_init__(self):
        finite(self.length, self.width)
        if self.length <= 0 or self.width <= 0:
            raise ValueError("Rectangle dimensions must be positive")

    @property
    def corners(self) -> tuple[Vec2, ...]:
        return tuple(
            self.pose.local_to_world(Vec2(x, y))
            for x, y in (
                (-self.length / 2, -self.width / 2),
                (self.length / 2, -self.width / 2),
                (self.length / 2, self.width / 2),
                (-self.length / 2, self.width / 2),
            )
        )

    @property
    def axes(self) -> tuple[Vec2, Vec2]:
        return (
            Vec2(cos(self.pose.yaw), sin(self.pose.yaw)),
            Vec2(-sin(self.pose.yaw), cos(self.pose.yaw)),
        )

    def contains(self, point: Vec2, margin: float = 0) -> bool:
        finite(margin)
        if margin < 0:
            raise ValueError("Margin cannot be negative")
        local = self.pose.world_to_local(point)
        return abs(local.x) <= self.length / 2 + margin and abs(local.y) <= self.width / 2 + margin

    def expanded(self, margin: float) -> "Rectangle":
        finite(margin)
        if margin < 0:
            raise ValueError("Expansion margin cannot be negative")
        return Rectangle(self.pose, self.length + 2 * margin, self.width + 2 * margin)

    def overlaps(self, other: "Rectangle") -> bool:
        """Separating axis test; touching counts as collision."""
        for axis in self.axes + other.axes:
            first = [corner.dot(axis) for corner in self.corners]
            second = [corner.dot(axis) for corner in other.corners]
            if max(first) < min(second) - 1e-12 or max(second) < min(first) - 1e-12:
                return False
        return True

    def aabb(self) -> tuple[float, float, float, float]:
        corners = self.corners
        return (
            min(p.x for p in corners),
            min(p.y for p in corners),
            max(p.x for p in corners),
            max(p.y for p in corners),
        )


def point_rectangle_distance(point: Vec2, box: Rectangle) -> float:
    local = box.pose.world_to_local(point)
    dx = max(abs(local.x) - box.length / 2, 0)
    dy = max(abs(local.y) - box.width / 2, 0)
    return Vec2(dx, dy).norm


def ray_rectangle_distance(origin: Vec2, direction: Vec2, box: Rectangle) -> float | None:
    """Slab intersection; returns zero if origin lies inside the rectangle."""
    unit = direction.normalized()
    local_origin = box.pose.world_to_local(origin)
    local_direction = unit.rotate(-box.pose.yaw)
    lower, upper = 0.0, float("inf")
    for value, speed, half in (
        (local_origin.x, local_direction.x, box.length / 2),
        (local_origin.y, local_direction.y, box.width / 2),
    ):
        if abs(speed) < 1e-12:
            if abs(value) > half:
                return None
            continue
        first = (-half - value) / speed
        second = (half - value) / speed
        lower = max(lower, min(first, second))
        upper = min(upper, max(first, second))
        if lower > upper:
            return None
    return lower
