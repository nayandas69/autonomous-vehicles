"""Annotated moving-box sequences for tracker regression experiments."""

from dataclasses import dataclass

import numpy as np

from autonomous_vehicles.evaluation.tracking import AnnotatedBox
from autonomous_vehicles.perception.boxes import Box, Detection


@dataclass(frozen=True)
class MovingBox:
    identifier: str
    initial: Box
    vx: float = 1.0
    vy: float = 0.0
    category: int = 2

    def __post_init__(self):
        if not self.identifier or not np.isfinite([self.vx, self.vy]).all() or self.category < 0:
            raise ValueError("Invalid moving box")

    def at(self, time: float) -> AnnotatedBox:
        if not np.isfinite(time) or time < 0:
            raise ValueError("Time must be finite and nonnegative")
        return AnnotatedBox(
            self.identifier, self.initial.translated(self.vx * time, self.vy * time), self.category
        )


@dataclass(frozen=True)
class SequenceFrame:
    time: float
    image: np.ndarray
    targets: tuple[AnnotatedBox, ...]
    detections: tuple[Detection, ...]


def moving_sequence(
    objects: tuple[MovingBox, ...],
    frames: int = 20,
    dt: float = 1.0,
    width: int = 128,
    height: int = 64,
    dropout: float = 0.0,
    noise: float = 0.0,
    seed: int = 42,
) -> tuple[SequenceFrame, ...]:
    if min(frames, width, height) < 1 or not np.isfinite(dt) or dt <= 0:
        raise ValueError("Invalid sequence dimensions or frame interval")
    if not 0 <= dropout <= 1 or not np.isfinite(noise) or noise < 0 or seed < 0:
        raise ValueError("Invalid sequence noise configuration")
    if len({obj.identifier for obj in objects}) != len(objects):
        raise ValueError("Object IDs must be unique")
    rng = np.random.default_rng(seed)
    result = []
    for index in range(frames):
        image = np.full((height, width, 3), 0.15, dtype=np.float32)
        targets, detections = [], []
        for obj in objects:
            target = obj.at(index * dt)
            box = target.box.clip(width, height)
            if box is None:
                continue
            targets.append(AnnotatedBox(target.identifier, box, target.category))
            image[int(box.y1) : int(np.ceil(box.y2)), int(box.x1) : int(np.ceil(box.x2))] = (
                0.85,
                0.2,
                0.1,
            )
            if rng.random() < dropout:
                continue
            dx, dy = rng.normal(0, noise, 2)
            noisy = box.translated(float(dx), float(dy)).clip(width, height)
            if noisy is not None:
                detections.append(Detection(noisy, 0.9, target.category))
        result.append(SequenceFrame(index * dt, image, tuple(targets), tuple(detections)))
    return tuple(result)
