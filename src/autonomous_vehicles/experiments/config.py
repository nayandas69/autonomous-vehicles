"""Strict JSON configuration loading for batch simulation experiments."""

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from autonomous_vehicles.simulation.scenarios import SCENARIOS


@dataclass(frozen=True)
class SimulationConfig:
    scenarios: tuple[str, ...] = ("lane_following", "winding_road", "stopped_vehicle")
    controllers: tuple[str, ...] = ("pursuit", "stanley")
    safety: bool = True
    repetitions: int = 1

    def __post_init__(self):
        if not self.scenarios or not self.controllers:
            raise ValueError("Scenarios and controllers cannot be empty")
        if any(name not in SCENARIOS for name in self.scenarios):
            raise ValueError("Unknown scenario in configuration")
        if any(name not in {"pursuit", "stanley"} for name in self.controllers):
            raise ValueError("Unknown controller in configuration")
        if len(set(self.scenarios)) != len(self.scenarios):
            raise ValueError("Scenario names cannot repeat")
        if len(set(self.controllers)) != len(self.controllers):
            raise ValueError("Controller names cannot repeat")
        if (
            type(self.safety) is not bool
            or type(self.repetitions) is not int
            or self.repetitions < 1
        ):
            raise ValueError("Safety must be boolean and repetitions a positive integer")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict):
        if not isinstance(payload, dict):
            raise ValueError("Configuration must be an object")
        unknown = set(payload) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
        converted = dict(payload)
        for name in ("scenarios", "controllers"):
            if name in converted:
                value = converted[name]
                if not isinstance(value, (list, tuple)) or any(
                    not isinstance(x, str) for x in value
                ):
                    raise ValueError(f"{name} must be a list of names")
                converted[name] = tuple(value)
        return cls(**converted)

    @classmethod
    def load(cls, path: Path):
        return cls.from_dict(json.loads(path.read_text()))
