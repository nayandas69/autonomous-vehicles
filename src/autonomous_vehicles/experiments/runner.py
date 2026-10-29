"""Batch evaluation with complete records and no implicit result overwrite."""

from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from autonomous_vehicles.control.driver import Driver
from autonomous_vehicles.evaluation.driving import aggregate_driving, driving_metrics
from autonomous_vehicles.experiments.artifacts import (
    atomic_json,
    environment_metadata,
    stable_identifier,
)
from autonomous_vehicles.experiments.config import SimulationConfig
from autonomous_vehicles.simulation.scenarios import make_scenario
from autonomous_vehicles.simulation.world import run_episode


def run_simulation_suite(config: SimulationConfig, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    records, all_metrics = [], []
    started = perf_counter()
    for scenario_name in config.scenarios:
        for controller in config.controllers:
            for repetition in range(config.repetitions):
                scenario = make_scenario(scenario_name)
                episode = run_episode(scenario, Driver(controller, safety=config.safety))
                metrics = driving_metrics(episode, scenario)
                all_metrics.append(metrics)
                record = {
                    "controller": controller,
                    "repetition": repetition,
                    "metrics": metrics.to_dict(),
                }
                records.append(record)
                name = f"{scenario_name}-{controller}-{repetition}.json"
                atomic_json(output / name, record)
    report = {
        "format_version": 1,
        "experiment_id": stable_identifier(config.to_dict()),
        "config": config.to_dict(),
        "environment": environment_metadata(),
        "elapsed_seconds": perf_counter() - started,
        "aggregate": aggregate_driving(tuple(all_metrics)),
        "records": records,
        "observation_model": "perfect_simulator_ground_truth",
    }
    atomic_json(output / "report.json", report)
    return report


def episode_table(episode) -> list[dict]:
    return [
        {
            "time": frame.state.time,
            "x": frame.state.pose.x,
            "y": frame.state.pose.y,
            "yaw": frame.state.pose.yaw,
            "speed": frame.state.speed,
            "acceleration": frame.state.acceleration,
            "steering": frame.state.steering,
            "target_speed": frame.target_speed,
            "lateral_error": frame.lateral_error,
            "collision": frame.collision,
            "off_road": frame.off_road,
            "emergency": frame.emergency,
            "actors": [asdict(actor) for actor in frame.actors],
        }
        for frame in episode.frames
    ]
