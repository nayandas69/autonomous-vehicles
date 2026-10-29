"""Research command handlers kept independent of argument parser construction."""

import json
from pathlib import Path

from autonomous_vehicles.control.driver import Driver
from autonomous_vehicles.engine import load_model
from autonomous_vehicles.evaluation.driving import driving_metrics
from autonomous_vehicles.evaluation.robustness import evaluate_corruptions, robustness_table
from autonomous_vehicles.experiments.artifacts import atomic_json, environment_metadata
from autonomous_vehicles.experiments.config import SimulationConfig
from autonomous_vehicles.experiments.runner import episode_table, run_simulation_suite
from autonomous_vehicles.perception.dataset import load_manifest, manifest_summary
from autonomous_vehicles.simulation.scenarios import SCENARIOS, make_scenario
from autonomous_vehicles.simulation.world import run_episode


def add_research_commands(commands):
    simulation = commands.add_parser("simulate", help="Run one synthetic closed-loop episode")
    simulation.add_argument("--scenario", choices=tuple(SCENARIOS), default="lane_following")
    simulation.add_argument("--controller", choices=("pursuit", "stanley"), default="pursuit")
    simulation.add_argument("--output", type=Path, required=True)
    simulation.add_argument("--without-safety", action="store_true")
    simulation.add_argument("--save-frames", action="store_true")
    benchmark = commands.add_parser("benchmark", help="Run a configured simulation suite")
    benchmark.add_argument("--config", type=Path)
    benchmark.add_argument("--output", type=Path, required=True)
    robustness = commands.add_parser("robustness", help="Evaluate seeded image corruptions")
    robustness.add_argument("--checkpoint", type=Path, required=True)
    robustness.add_argument("--samples", type=int, default=32)
    robustness.add_argument("--output", type=Path, required=True)
    dataset = commands.add_parser("inspect-dataset", help="Validate a local data manifest")
    dataset.add_argument("--manifest", type=Path, required=True)
    commands.add_parser("environment", help="Print tested environment identifiers")
    commands.add_parser("scenarios", help="List available synthetic scenarios")


def research_command(args):
    if args.command == "simulate":
        return simulate(args)
    if args.command == "benchmark":
        config = SimulationConfig.load(args.config) if args.config else SimulationConfig()
        return run_simulation_suite(config, args.output)
    if args.command == "robustness":
        model, config = load_model(args.checkpoint)
        results = evaluate_corruptions(model, args.samples, config.size, config.seed)
        report = {"format_version": 1, "split": "test", "results": robustness_table(results)}
        atomic_json(args.output, report)
        return report
    if args.command == "inspect-dataset":
        return manifest_summary(load_manifest(args.manifest))
    if args.command == "environment":
        return environment_metadata()
    if args.command == "scenarios":
        return {"scenarios": list(SCENARIOS)}
    raise ValueError(f"Unsupported command {args.command!r}")


def simulate(args):
    # Reserve a run directory before expensive execution.
    args.output.mkdir(parents=True, exist_ok=False)
    scenario = make_scenario(args.scenario)
    episode = run_episode(scenario, Driver(args.controller, safety=not args.without_safety))
    report = {
        "format_version": 1,
        "scenario": args.scenario,
        "controller": args.controller,
        "safety": not args.without_safety,
        "observation_model": "perfect_simulator_ground_truth",
        "environment": environment_metadata(),
        "metrics": driving_metrics(episode, scenario).to_dict(),
    }
    atomic_json(args.output / "report.json", report)
    if args.save_frames:
        atomic_json(args.output / "frames.json", episode_table(episode))
    return report


def render_result(result) -> str:
    return json.dumps(result, indent=2, allow_nan=False)
