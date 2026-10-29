"""Command-line entry points for reproducible experiments."""

import argparse
import json
from pathlib import Path

from torch.utils.data import DataLoader

from autonomous_vehicles.data import RoadScenes
from autonomous_vehicles.engine import TrainConfig, load_model, train
from autonomous_vehicles.metrics import evaluate
from autonomous_vehicles.research_cli import add_research_commands, research_command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--output", type=Path, required=True)
    for name in ("epochs", "samples", "batch_size", "size", "seed"):
        training.add_argument(
            "--" + name.replace("_", "-"), type=int, default=getattr(TrainConfig(), name)
        )
    testing = commands.add_parser("evaluate")
    testing.add_argument("--checkpoint", type=Path, required=True)
    testing.add_argument("--samples", type=int, default=64)
    add_research_commands(commands)
    args = parser.parse_args()
    try:
        if args.command == "train":
            config = TrainConfig(
                **{k: getattr(args, k) for k in ("epochs", "samples", "batch_size", "size", "seed")}
            )
            result = train(config, args.output)
        elif args.command == "evaluate":
            model, config = load_model(args.checkpoint)
            loader = DataLoader(
                RoadScenes(args.samples, config.size, config.seed, "test"),
                batch_size=config.batch_size,
            )
            result = {"split": "test", "samples": args.samples, **evaluate(model, loader)}
        else:
            result = research_command(args)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(result, indent=2))
