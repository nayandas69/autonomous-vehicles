"""Named, deterministic scenarios for controller and safety research.

All participants are synthetic. These scenarios are small regression fixtures,
not a validated automotive scenario corpus.
"""

from dataclasses import dataclass
from math import pi

from autonomous_vehicles.simulation.actors import (
    ActorState,
    LinearActor,
    RoadActor,
    static_obstacle,
)
from autonomous_vehicles.simulation.dynamics import VehicleParameters, VehicleState
from autonomous_vehicles.simulation.geometry import Pose2, finite
from autonomous_vehicles.simulation.roads import Road, circular_road, curved_road, straight_road


@dataclass(frozen=True)
class Scenario:
    name: str
    road: Road
    initial: VehicleState
    actors: tuple[LinearActor | RoadActor, ...] = ()
    duration: float = 15.0
    dt: float = 0.1
    target_speed: float = 8.0
    parameters: VehicleParameters = VehicleParameters()

    def __post_init__(self):
        finite(self.duration, self.dt, self.target_speed)
        if not self.name or self.duration <= 0 or not 0 < self.dt <= 0.5:
            raise ValueError("Scenario requires a name, positive duration and dt in (0, 0.5]")
        if not 0 <= self.target_speed <= self.parameters.max_speed:
            raise ValueError("Target speed outside vehicle limits")
        if self.initial.time != 0:
            raise ValueError("Scenario initial time must be zero")
        identifiers = [actor.at(0).identifier for actor in self.actors]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Actor identifiers must be unique")

    def actors_at(self, time: float) -> tuple[ActorState, ...]:
        return tuple(actor.at(time) for actor in self.actors)


def lane_following() -> Scenario:
    road = straight_road()
    return Scenario("lane_following", road, VehicleState(Pose2(3, 0.8), speed=2))


def winding_road() -> Scenario:
    road = curved_road(amplitude=5, wavelength=140)
    return Scenario("winding_road", road, VehicleState(road.pose_at(3), speed=2), target_speed=6)


def gentle_turn() -> Scenario:
    road = circular_road(radius=60)
    return Scenario(
        "gentle_turn", road, VehicleState(road.pose_at(3), speed=2), duration=12, target_speed=5
    )


def stopped_vehicle() -> Scenario:
    road = straight_road()
    return Scenario(
        "stopped_vehicle",
        road,
        VehicleState(Pose2(3, 0), speed=6),
        (static_obstacle("stopped", 45, 0, 4.5, 1.8),),
    )


def lead_vehicle() -> Scenario:
    road = straight_road()
    return Scenario(
        "lead_vehicle",
        road,
        VehicleState(Pose2(3, 0), speed=6),
        (RoadActor("lead", road, 25, 3),),
        target_speed=8,
    )


def crossing_pedestrian() -> Scenario:
    road = straight_road()
    pedestrian = LinearActor(
        ActorState("walker", Pose2(35, -5, pi / 2), 1.2, 0.6, 0.6, "pedestrian"),
        start_time=1,
    )
    return Scenario(
        "crossing_pedestrian",
        road,
        VehicleState(Pose2(3, 0), speed=5),
        (pedestrian,),
        target_speed=6,
    )


def adjacent_traffic() -> Scenario:
    road = straight_road(width=10)
    return Scenario(
        "adjacent_traffic",
        road,
        VehicleState(Pose2(3, 0), speed=5),
        (RoadActor("adjacent", road, 20, 4, lateral=3.5),),
        target_speed=7,
    )


SCENARIOS = {
    "lane_following": lane_following,
    "winding_road": winding_road,
    "gentle_turn": gentle_turn,
    "stopped_vehicle": stopped_vehicle,
    "lead_vehicle": lead_vehicle,
    "crossing_pedestrian": crossing_pedestrian,
    "adjacent_traffic": adjacent_traffic,
}


def make_scenario(name: str) -> Scenario:
    if name not in SCENARIOS:
        raise ValueError(f"Unknown scenario {name!r}; choose from {', '.join(SCENARIOS)}")
    return SCENARIOS[name]()
