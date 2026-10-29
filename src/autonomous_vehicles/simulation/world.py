"""Fixed-step closed-loop simulation with substep collision detection.

The controller sees perfect actor ground truth. Perception experiments are
separate and their predictions are not used for vehicle control.
"""

from dataclasses import dataclass
from math import ceil

from autonomous_vehicles.control.driver import Driver
from autonomous_vehicles.simulation.actors import ActorState
from autonomous_vehicles.simulation.dynamics import Control, VehicleState, bicycle_step
from autonomous_vehicles.simulation.scenarios import Scenario


@dataclass(frozen=True)
class Frame:
    state: VehicleState
    actors: tuple[ActorState, ...]
    control: Control
    lateral_error: float
    target_speed: float
    emergency: bool
    collision: bool
    off_road: bool


@dataclass(frozen=True)
class Episode:
    scenario_name: str
    frames: tuple[Frame, ...]
    termination: str
    dt: float

    @property
    def final(self) -> VehicleState:
        if not self.frames:
            raise ValueError("Episode has no frames")
        return self.frames[-1].state

    @property
    def collided(self) -> bool:
        return any(frame.collision for frame in self.frames)

    @property
    def left_road(self) -> bool:
        return any(frame.off_road for frame in self.frames)


def collision_at(state: VehicleState, actors: tuple[ActorState, ...], scenario: Scenario) -> bool:
    body = state.footprint(scenario.parameters)
    return any(body.overlaps(actor.footprint) for actor in actors)


def off_road_at(state: VehicleState, scenario: Scenario) -> bool:
    # Rear axle road projection defines progress; corners enforce body containment.
    corners = state.footprint(scenario.parameters).corners
    return any(not scenario.road.contains(corner) for corner in corners)


def run_episode(scenario: Scenario, driver: Driver | None = None) -> Episode:
    driver = Driver() if driver is None else driver
    driver.reset()
    state = scenario.initial
    frames = []
    termination = "duration"
    steps = int(ceil(scenario.duration / scenario.dt))
    for index in range(steps + 1):
        actors = scenario.actors_at(state.time)
        output = driver.command(
            state,
            scenario.road,
            actors,
            scenario.target_speed,
            scenario.dt,
            scenario.parameters,
        )
        collision = collision_at(state, actors, scenario)
        off_road = off_road_at(state, scenario)
        frames.append(
            Frame(
                state,
                actors,
                output.control,
                output.lateral_error,
                output.decision.target_speed,
                output.decision.emergency,
                collision,
                off_road,
            )
        )
        if collision:
            termination = "collision"
            break
        if off_road:
            termination = "off_road"
            break
        if scenario.road.project(state.pose.position).s >= scenario.road.length - 5:
            termination = "route_complete"
            break
        if index == steps:
            break
        remaining = scenario.duration - state.time
        dt = min(scenario.dt, remaining)
        if dt <= 1e-9:
            break
        # Substeps limit swept-motion misses; these are still discrete checks.
        substeps = max(1, int(ceil(dt / 0.02)))
        for _ in range(substeps):
            state = bicycle_step(state, output.control, dt / substeps, scenario.parameters)
            if collision_at(state, scenario.actors_at(state.time), scenario):
                break
    return Episode(scenario.name, tuple(frames), termination, scenario.dt)
