"""Composition of lateral steering, speed regulation, and stopping envelope."""

from dataclasses import dataclass

from autonomous_vehicles.control.lateral import PurePursuit, Stanley
from autonomous_vehicles.control.pid import PIDConfig, SpeedController
from autonomous_vehicles.control.safety import SafetyDecision, StoppingEnvelope
from autonomous_vehicles.simulation.actors import ActorState
from autonomous_vehicles.simulation.dynamics import Control, VehicleParameters, VehicleState
from autonomous_vehicles.simulation.roads import Road


@dataclass(frozen=True)
class DriverOutput:
    control: Control
    decision: SafetyDecision
    lateral_error: float


class Driver:
    def __init__(self, lateral: str = "pursuit", safety: bool = True):
        if lateral not in {"pursuit", "stanley"}:
            raise ValueError("Choose pursuit or stanley")
        self.lateral_name = lateral
        self.lateral = PurePursuit() if lateral == "pursuit" else Stanley()
        self.use_safety = safety
        self.envelope = StoppingEnvelope()
        self.speed = SpeedController()
        self._parameters = None

    def reset(self) -> None:
        self.speed.reset()
        self._parameters = None

    def command(
        self,
        state: VehicleState,
        road: Road,
        actors: tuple[ActorState, ...],
        target_speed: float,
        dt: float,
        parameters: VehicleParameters,
    ) -> DriverOutput:
        if self._parameters != parameters:
            self.speed = SpeedController(
                PIDConfig(
                    lower=-parameters.max_deceleration,
                    upper=parameters.max_acceleration,
                )
            )
            self._parameters = parameters
        desired = min(target_speed, road.speed_limit, parameters.max_speed)
        decision = (
            self.envelope.decide(state, actors, desired, parameters)
            if self.use_safety
            else SafetyDecision(desired, False, None, None, "disabled")
        )
        acceleration = self.speed.command(decision.target_speed, state.speed, dt)
        if decision.emergency:
            acceleration = -parameters.max_deceleration
            self.speed.reset()
        steering = self.lateral.command(state, road, parameters)
        return DriverOutput(
            Control(acceleration, steering),
            decision,
            road.project(state.pose.position).lateral,
        )
