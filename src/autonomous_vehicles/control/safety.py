"""Conservative longitudinal stopping envelope using simulator ground truth.

This is an experimental rule, not a certified collision avoidance system.
Actors outside the lateral corridor are ignored. Crossing actors can therefore
be missed until they enter the corridor; scenario results expose this limitation.
"""

from dataclasses import dataclass
from math import sqrt

from autonomous_vehicles.simulation.actors import ActorState
from autonomous_vehicles.simulation.dynamics import (
    VehicleParameters,
    VehicleState,
    stopping_distance,
)
from autonomous_vehicles.simulation.geometry import finite


@dataclass(frozen=True)
class SafetyConfig:
    minimum_gap: float = 3.0
    time_headway: float = 1.5
    reaction_time: float = 0.3
    comfortable_deceleration: float = 3.0
    corridor_margin: float = 0.5

    def __post_init__(self):
        finite(*vars(self).values())
        if min(self.minimum_gap, self.time_headway, self.reaction_time, self.corridor_margin) < 0:
            raise ValueError("Safety margins must be nonnegative")
        if self.comfortable_deceleration <= 0:
            raise ValueError("Deceleration must be positive")


@dataclass(frozen=True)
class SafetyDecision:
    target_speed: float
    emergency: bool
    nearest_gap: float | None
    actor_id: str | None
    reason: str


class StoppingEnvelope:
    def __init__(self, config: SafetyConfig | None = None):
        self.config = SafetyConfig() if config is None else config

    def decide(
        self,
        state: VehicleState,
        actors: tuple[ActorState, ...],
        desired_speed: float,
        parameters: VehicleParameters,
    ) -> SafetyDecision:
        finite(desired_speed)
        if desired_speed < 0:
            raise ValueError("Desired speed cannot be negative")
        nearest = None
        nearest_actor = None
        for actor in actors:
            local = state.pose.world_to_local(actor.pose.position)
            half_diagonal = (actor.length**2 + actor.width**2) ** 0.5 / 2
            corridor = parameters.width / 2 + self.config.corridor_margin + half_diagonal
            if local.x < 0 or abs(local.y) > corridor:
                continue
            gap = local.x - (parameters.length - parameters.rear_overhang) - half_diagonal
            if nearest is None or gap < nearest:
                nearest, nearest_actor = gap, actor
        if nearest is None:
            return SafetyDecision(desired_speed, False, None, None, "clear")
        available = max(0, nearest - self.config.minimum_gap)
        # Solve v*t + v²/(2a) <= available for a stationary obstacle.
        a = min(self.config.comfortable_deceleration, parameters.max_deceleration)
        reaction = self.config.reaction_time + self.config.time_headway
        allowed = max(0, sqrt((a * reaction) ** 2 + 2 * a * available) - a * reaction)
        emergency_distance = stopping_distance(
            state.speed, parameters.max_deceleration, self.config.reaction_time
        )
        emergency = nearest <= emergency_distance + self.config.minimum_gap
        return SafetyDecision(
            min(desired_speed, allowed),
            emergency,
            nearest,
            nearest_actor.identifier,
            "emergency" if emergency else "following_gap",
        )
