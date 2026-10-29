"""Integration of actor scripts, controllers, collision events and reports."""

from dataclasses import replace
from math import pi

import pytest

from autonomous_vehicles.control.driver import Driver
from autonomous_vehicles.evaluation.driving import aggregate_driving, driving_metrics
from autonomous_vehicles.experiments.config import SimulationConfig
from autonomous_vehicles.experiments.runner import episode_table, run_simulation_suite
from autonomous_vehicles.simulation.actors import (
    ActorState,
    LinearActor,
    RoadActor,
    static_obstacle,
)
from autonomous_vehicles.simulation.dynamics import VehicleState
from autonomous_vehicles.simulation.geometry import Pose2
from autonomous_vehicles.simulation.scenarios import SCENARIOS, Scenario, make_scenario
from autonomous_vehicles.simulation.world import collision_at, run_episode


def test_delayed_linear_actor_and_road_actor_stop_at_end():
    linear = LinearActor(ActorState("walker", Pose2(0, 0, pi / 2), 2), start_time=1)
    assert linear.at(0).speed == 0
    assert linear.at(0).pose.y == 0
    assert linear.at(2).pose.y == pytest.approx(2)
    road = make_scenario("lane_following").road
    actor = RoadActor("lead", road, road.length - 1, 2)
    assert actor.at(2).pose.x == road.length
    assert actor.at(2).speed == 0


@pytest.mark.parametrize("name", tuple(SCENARIOS))
def test_scenarios_have_unique_actors_and_valid_initial_states(name):
    scenario = make_scenario(name)
    actors = scenario.actors_at(0)
    assert len({actor.identifier for actor in actors}) == len(actors)
    assert scenario.road.contains(scenario.initial.pose.position)
    assert not collision_at(scenario.initial, actors, scenario)


@pytest.mark.parametrize("controller", ["pursuit", "stanley"])
def test_lane_following_converges_without_events(controller):
    scenario = replace(make_scenario("lane_following"), duration=5)
    episode = run_episode(scenario, Driver(controller))
    assert not episode.collided
    assert not episode.left_road
    assert abs(episode.frames[-1].lateral_error) < abs(episode.frames[0].lateral_error)
    assert episode.final.time == pytest.approx(5)
    metrics = driving_metrics(episode, scenario)
    assert metrics.progress_meters > 10
    assert metrics.duration_seconds == pytest.approx(5)
    assert metrics.distance_meters >= metrics.progress_meters - 0.1


def test_stopped_vehicle_triggers_braking_and_prevents_collision():
    scenario = make_scenario("stopped_vehicle")
    episode = run_episode(scenario)
    assert not episode.collided
    assert episode.final.speed < 1
    assert episode.final.pose.x < 42
    assert episode.frames[-1].target_speed < scenario.target_speed


def test_initial_collision_stops_immediately():
    original = make_scenario("lane_following")
    obstacle = static_obstacle("overlap", 4, 0, 4, 2)
    scenario = replace(original, actors=(obstacle,))
    episode = run_episode(scenario)
    assert episode.termination == "collision"
    assert episode.collided
    assert len(episode.frames) == 1


def test_initial_off_road_is_reported():
    original = make_scenario("lane_following")
    scenario = replace(original, initial=VehicleState(Pose2(5, 10)))
    episode = run_episode(scenario)
    assert episode.termination == "off_road"
    assert episode.left_road


def test_episode_table_and_aggregate_are_json_ready():
    scenario = replace(make_scenario("lane_following"), duration=0.2)
    episode = run_episode(scenario)
    rows = episode_table(episode)
    assert rows[0]["time"] == 0
    assert rows[-1]["time"] == pytest.approx(0.2)
    metrics = driving_metrics(episode, scenario)
    aggregate = aggregate_driving((metrics,))
    assert aggregate["episodes"] == 1
    assert aggregate["collision_rate"] == 0


def test_suite_writes_each_record_and_refuses_overwrite(tmp_path):
    config = SimulationConfig(scenarios=("lane_following",), controllers=("pursuit",))
    report = run_simulation_suite(config, tmp_path / "run")
    assert report["format_version"] == 1
    assert report["observation_model"] == "perfect_simulator_ground_truth"
    assert len(report["records"]) == 1
    assert (tmp_path / "run/report.json").is_file()
    with pytest.raises(FileExistsError):
        run_simulation_suite(config, tmp_path / "run")


def test_invalid_scenarios_and_actor_contracts():
    with pytest.raises(ValueError):
        make_scenario("unknown")
    with pytest.raises(ValueError):
        ActorState("", Pose2(), 0)
    with pytest.raises(ValueError):
        Scenario("bad", make_scenario("lane_following").road, VehicleState(), dt=0)
