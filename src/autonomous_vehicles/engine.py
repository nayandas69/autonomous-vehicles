"""CPU experiments with deterministic training and validation-only model selection."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from autonomous_vehicles.data import CLASS_NAMES, RoadScenes
from autonomous_vehicles.metrics import evaluate
from autonomous_vehicles.model import RoadNet


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 3
    samples: int = 128
    batch_size: int = 16
    size: int = 64
    seed: int = 42
    learning_rate: float = 0.001

    def __post_init__(self):
        if min(self.epochs, self.samples, self.batch_size) < 1 or self.size < 32 or self.seed < 0:
            raise ValueError("Counts must be positive, size >= 32, seed >= 0")
        if not 0 < self.learning_rate < float("inf"):
            raise ValueError("learning_rate must be finite and positive")


def train(config: TrainConfig, output: Path) -> dict:
    # Refuse to overwrite an existing run, including partially completed runs.
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    model = RoadNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    generator = torch.Generator().manual_seed(config.seed)
    training = DataLoader(
        RoadScenes(config.samples, config.size, config.seed),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    validation = DataLoader(
        RoadScenes(max(16, config.samples // 4), config.size, config.seed, "validation"),
        batch_size=config.batch_size,
    )
    history, best = [], -1.0
    for epoch in range(config.epochs):
        model.train()
        total = 0.0
        for image, target in training:
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(image), target)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(image)
        metrics = evaluate(model, validation)
        history.append({"epoch": epoch + 1, "train_loss": total / config.samples, **metrics})
        if metrics["mean_iou"] > best:
            best = metrics["mean_iou"]
            torch.save(
                {"format_version": 1, "config": asdict(config), "state_dict": model.state_dict()},
                output / "model.pt",
            )
    report = {
        "config": asdict(config),
        "torch_version": str(torch.__version__),
        "classes": CLASS_NAMES,
        "history": history,
        "best_validation_iou": best,
    }
    (output / "training.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def load_model(checkpoint: Path):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    config = TrainConfig(**payload["config"])
    model = RoadNet()
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, config
