"""Pure pursuit and Stanley steering controllers over metric reference roads."""

from dataclasses import dataclass
from math import atan, atan2, hypot, sin

from autonomous_vehicles.simulation.dynamics import VehicleParameters, VehicleState
from autonomous_vehicles.simulation.geometry import Vec2, angle_difference, clamp, finite
from autonomous_vehicles.simulation.roads import Road


@dataclass(frozen=True)
class PursuitConfig:
    minimum_lookahead: float = 3.0
    speed_gain: float = 0.6
    maximum_lookahead: float = 15.0

    def __post_init__(self):
        finite(*vars(self).values())
        if self.minimum_lookahead <= 0 or self.speed_gain < 0:
            raise ValueError("Lookahead must be positive and speed gain nonnegative")
        if self.maximum_lookahead < self.minimum_lookahead:
            raise ValueError("Maximum lookahead is smaller than minimum")


class PurePursuit:
    def __init__(self, config: PursuitConfig | None = None):
        self.config = PursuitConfig() if config is None else config

    def command(self, state: VehicleState, road: Road, parameters: VehicleParameters) -> float:
        projection = road.project(state.pose.position)
        distance = clamp(
            self.config.minimum_lookahead + self.config.speed_gain * state.speed,
            self.config.minimum_lookahead,
            self.config.maximum_lookahead,
        )
        target = road.pose_at(projection.s + distance)
        local = state.pose.world_to_local(target.position)
        squared_distance = local.x * local.x + local.y * local.y
        if squared_distance <= 1e-12:
            return 0.0
        steering = atan2(2 * parameters.wheelbase * local.y, squared_distance)
        return clamp(steering, -parameters.max_steering, parameters.max_steering)


@dataclass(frozen=True)
class StanleyConfig:
    cross_track_gain: float = 1.0
    softening_speed: float = 1.0
    heading_gain: float = 1.0

    def __post_init__(self):
        finite(*vars(self).values())
        if self.cross_track_gain < 0 or self.heading_gain < 0 or self.softening_speed <= 0:
            raise ValueError("Invalid Stanley gains")


class Stanley:
    def __init__(self, config: StanleyConfig | None = None):
        self.config = StanleyConfig() if config is None else config

    def command(self, state: VehicleState, road: Road, parameters: VehicleParameters) -> float:
        front = state.pose.local_to_world(Vec2(parameters.wheelbase, 0))
        projection = road.project(front)
        heading_error = angle_difference(projection.pose.yaw, state.pose.yaw)
        correction = atan(
            -self.config.cross_track_gain
            * projection.lateral
            / (state.speed + self.config.softening_speed)
        )
        return clamp(
            self.config.heading_gain * heading_error + correction,
            -parameters.max_steering,
            parameters.max_steering,
        )


def pursuit_curvature(target_x: float, target_y: float) -> float:
    finite(target_x, target_y)
    distance = hypot(target_x, target_y)
    if distance < 1e-9:
        return 0.0
    return 2 * sin(atan2(target_y, target_x)) / distance
