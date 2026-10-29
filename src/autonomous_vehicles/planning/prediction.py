"""Constant-velocity actor prediction and sampled trajectory collision checks."""

from dataclasses import dataclass

from autonomous_vehicles.planning.path import Trajectory
from autonomous_vehicles.simulation.actors import ActorState
from autonomous_vehicles.simulation.dynamics import VehicleParameters, VehicleState
from autonomous_vehicles.simulation.geometry import Pose2, finite


@dataclass(frozen=True)
class CollisionRisk:
    collision: bool
    earliest_time: float | None
    actor_id: str | None
    checked_samples: int


def predict_actor(actor: ActorState, horizon: float) -> ActorState:
    finite(horizon)
    if horizon < 0:
        raise ValueError("Prediction horizon must be nonnegative")
    position = actor.pose.position + actor.velocity * horizon
    return ActorState(
        actor.identifier,
        Pose2(position.x, position.y, actor.pose.yaw),
        actor.speed,
        actor.length,
        actor.width,
        actor.category,
    )


def check_trajectory(
    trajectory: Trajectory,
    actors: tuple[ActorState, ...],
    parameters: VehicleParameters | None = None,
    margin: float = 0.2,
) -> CollisionRisk:
    parameters = VehicleParameters() if parameters is None else parameters
    finite(margin)
    if margin < 0:
        raise ValueError("Collision margin cannot be negative")
    start = trajectory.points[0].time
    for index, point in enumerate(trajectory.points):
        vehicle = VehicleState(point.pose, point.speed)
        body = vehicle.footprint(parameters).expanded(margin)
        for actor in actors:
            predicted = predict_actor(actor, point.time - start)
            if body.overlaps(predicted.footprint):
                return CollisionRisk(True, point.time - start, actor.identifier, index + 1)
    return CollisionRisk(False, None, None, len(trajectory.points))


def time_to_closest_approach(
    first: ActorState, second: ActorState, horizon: float = 10
) -> tuple[float, float]:
    finite(horizon)
    if horizon < 0:
        raise ValueError("Horizon cannot be negative")
    relative_position = second.pose.position - first.pose.position
    relative_velocity = second.velocity - first.velocity
    squared_speed = relative_velocity.dot(relative_velocity)
    time = (
        0.0
        if squared_speed < 1e-12
        else max(
            0.0,
            min(horizon, -relative_position.dot(relative_velocity) / squared_speed),
        )
    )
    return time, (relative_position + relative_velocity * time).norm
