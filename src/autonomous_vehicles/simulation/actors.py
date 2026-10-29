"""Scripted traffic participants with time-indexed ground-truth poses."""

from dataclasses import dataclass
from math import cos, sin

from autonomous_vehicles.simulation.geometry import Pose2, Vec2, finite
from autonomous_vehicles.simulation.roads import Road
from autonomous_vehicles.simulation.shapes import Rectangle


@dataclass(frozen=True)
class ActorState:
    identifier: str
    pose: Pose2
    speed: float
    length: float = 4.5
    width: float = 1.8
    category: str = "vehicle"

    def __post_init__(self):
        finite(self.speed, self.length, self.width)
        if not self.identifier or self.speed < 0 or min(self.length, self.width) <= 0:
            raise ValueError("Actor needs an ID, nonnegative speed and positive dimensions")
        if self.category not in {"vehicle", "pedestrian", "obstacle"}:
            raise ValueError("Unknown actor category")

    @property
    def footprint(self) -> Rectangle:
        return Rectangle(self.pose, self.length, self.width)

    @property
    def velocity(self) -> Vec2:
        return Vec2(self.speed * cos(self.pose.yaw), self.speed * sin(self.pose.yaw))


@dataclass(frozen=True)
class LinearActor:
    initial: ActorState
    start_time: float = 0.0

    def __post_init__(self):
        finite(self.start_time)
        if self.start_time < 0:
            raise ValueError("Start time cannot be negative")

    def at(self, time: float) -> ActorState:
        finite(time)
        if time < 0:
            raise ValueError("Time cannot be negative")
        elapsed = max(0, time - self.start_time)
        point = self.initial.pose.position + self.initial.velocity * elapsed
        return ActorState(
            self.initial.identifier,
            Pose2(point.x, point.y, self.initial.pose.yaw),
            self.initial.speed if time >= self.start_time else 0,
            self.initial.length,
            self.initial.width,
            self.initial.category,
        )


@dataclass(frozen=True)
class RoadActor:
    identifier: str
    road: Road
    initial_s: float
    speed: float
    lateral: float = 0.0
    length: float = 4.5
    width: float = 1.8

    def __post_init__(self):
        finite(self.initial_s, self.speed, self.lateral, self.length, self.width)
        if not self.identifier or self.initial_s < 0 or self.speed < 0:
            raise ValueError("Invalid road actor")
        if self.initial_s > self.road.length or min(self.length, self.width) <= 0:
            raise ValueError("Actor must begin on road with positive dimensions")

    def at(self, time: float) -> ActorState:
        finite(time)
        if time < 0:
            raise ValueError("Time cannot be negative")
        s = min(self.road.length, self.initial_s + time * self.speed)
        return ActorState(
            self.identifier,
            self.road.pose_at(s, self.lateral),
            self.speed if s < self.road.length else 0,
            self.length,
            self.width,
        )


def static_obstacle(identifier: str, x: float, y: float, length: float = 3, width: float = 2):
    return LinearActor(ActorState(identifier, Pose2(x, y), 0, length, width, "obstacle"))
