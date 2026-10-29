"""Forward-only kinematic bicycle dynamics with bounded actuator rates.

Pose is at the rear axle. Geometry queries account for the body-center offset.
Integration uses constant controls and midpoint heading per fixed time step.
"""

from dataclasses import dataclass
from math import atan, cos, sin, tan

from autonomous_vehicles.simulation.geometry import Pose2, clamp, finite, wrap_angle
from autonomous_vehicles.simulation.shapes import Rectangle


@dataclass(frozen=True)
class VehicleParameters:
    wheelbase: float = 2.7
    length: float = 4.5
    width: float = 1.8
    rear_overhang: float = 0.9
    max_speed: float = 25.0
    max_acceleration: float = 3.0
    max_deceleration: float = 7.0
    max_steering: float = 0.55
    max_steering_rate: float = 0.8

    def __post_init__(self):
        values = vars(self)
        finite(*values.values())
        if any(value <= 0 for value in values.values()):
            raise ValueError("Vehicle parameters must be positive")
        if self.length < self.wheelbase + self.rear_overhang:
            raise ValueError("Body must contain both axles and rear overhang")
        if self.max_steering >= 1.4:
            raise ValueError("Steering limit must be below 1.4 radians")

    @property
    def center_offset(self) -> float:
        return self.length / 2 - self.rear_overhang


@dataclass(frozen=True)
class VehicleState:
    pose: Pose2 = Pose2()
    speed: float = 0.0
    steering: float = 0.0
    acceleration: float = 0.0
    time: float = 0.0

    def __post_init__(self):
        finite(self.speed, self.steering, self.acceleration, self.time)
        if self.speed < 0 or self.time < 0:
            raise ValueError("Speed and time must be nonnegative")

    def footprint(self, parameters: VehicleParameters) -> Rectangle:
        return Rectangle(
            self.pose.compose(Pose2(parameters.center_offset, 0, 0)),
            parameters.length,
            parameters.width,
        )


@dataclass(frozen=True)
class Control:
    acceleration: float = 0.0
    steering: float = 0.0

    def __post_init__(self):
        finite(self.acceleration, self.steering)

    def bounded(self, parameters: VehicleParameters) -> "Control":
        return Control(
            clamp(self.acceleration, -parameters.max_deceleration, parameters.max_acceleration),
            clamp(self.steering, -parameters.max_steering, parameters.max_steering),
        )


def bicycle_step(
    state: VehicleState,
    control: Control,
    dt: float,
    parameters: VehicleParameters | None = None,
) -> VehicleState:
    parameters = VehicleParameters() if parameters is None else parameters
    finite(dt)
    if not 0 < dt <= 0.5:
        raise ValueError("Time step must lie in (0, 0.5] seconds")
    control = control.bounded(parameters)
    rate = parameters.max_steering_rate * dt
    steering = clamp(control.steering, state.steering - rate, state.steering + rate)
    steering = clamp(steering, -parameters.max_steering, parameters.max_steering)
    speed = clamp(state.speed + control.acceleration * dt, 0, parameters.max_speed)
    average_speed = (state.speed + speed) / 2
    curvature = tan(steering) / parameters.wheelbase
    yaw_change = average_speed * curvature * dt
    midpoint_heading = state.pose.yaw + yaw_change / 2
    pose = Pose2(
        state.pose.x + average_speed * cos(midpoint_heading) * dt,
        state.pose.y + average_speed * sin(midpoint_heading) * dt,
        wrap_angle(state.pose.yaw + yaw_change),
    )
    return VehicleState(
        pose,
        speed,
        steering,
        (speed - state.speed) / dt,
        state.time + dt,
    )


def steering_for_curvature(curvature: float, parameters: VehicleParameters) -> float:
    finite(curvature)
    return clamp(
        atan(parameters.wheelbase * curvature), -parameters.max_steering, parameters.max_steering
    )


def stopping_distance(speed: float, deceleration: float, reaction_time: float = 0.0) -> float:
    finite(speed, deceleration, reaction_time)
    if speed < 0 or deceleration <= 0 or reaction_time < 0:
        raise ValueError("Require speed >= 0, deceleration > 0 and reaction_time >= 0")
    return speed * reaction_time + speed * speed / (2 * deceleration)
