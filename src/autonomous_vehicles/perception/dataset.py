"""Manifest-based local NumPy datasets with explicit split and label validation.

Manifest paths are relative to the manifest directory and cannot escape it.
Files must already exist; this module never downloads datasets.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SampleRecord:
    identifier: str
    scene: str
    split: str
    image: str
    mask: str

    def __post_init__(self):
        if not all((self.identifier, self.scene, self.image, self.mask)):
            raise ValueError("Record fields cannot be empty")
        if self.split not in {"train", "validation", "test"}:
            raise ValueError("Unknown split")


def confined_path(root: Path, relative: str) -> Path:
    if Path(relative).is_absolute():
        raise ValueError("Dataset paths must be relative")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("Dataset path escapes manifest directory")
    return resolved


def load_manifest(path: Path) -> tuple[SampleRecord, ...]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or payload.get("format_version") != 1:
        raise ValueError("Unsupported dataset manifest format")
    if not isinstance(payload.get("samples"), list) or not payload["samples"]:
        raise ValueError("Manifest needs a nonempty samples list")
    records = tuple(SampleRecord(**record) for record in payload["samples"])
    identifiers, scenes, images = set(), {}, {}
    for record in records:
        if record.identifier in identifiers:
            raise ValueError(f"Duplicate sample ID: {record.identifier}")
        identifiers.add(record.identifier)
        if record.scene in scenes and scenes[record.scene] != record.split:
            raise ValueError(f"Scene leaks across splits: {record.scene}")
        scenes[record.scene] = record.split
        for field in (record.image, record.mask):
            if not confined_path(path.parent, field).is_file():
                raise ValueError(f"Dataset file missing: {field}")
        image = confined_path(path.parent, record.image)
        if image in images:
            raise ValueError("An image file appears in multiple records")
        images[image] = record.identifier
    return records


class ManifestDataset(Dataset):
    def __init__(self, manifest: Path, split: str, classes: int = 3):
        if split not in {"train", "validation", "test"} or classes < 2:
            raise ValueError("Invalid split or class count")
        self.root = manifest.resolve().parent
        self.records = tuple(record for record in load_manifest(manifest) if record.split == split)
        self.classes = classes
        if not self.records:
            raise ValueError(f"No samples in split {split}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = np.load(confined_path(self.root, record.image), allow_pickle=False)
        mask = np.load(confined_path(self.root, record.mask), allow_pickle=False)
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
            raise ValueError(f"{record.identifier}: image must be HWC uint8 RGB")
        if mask.shape != image.shape[:2] or not np.issubdtype(mask.dtype, np.integer):
            raise ValueError(f"{record.identifier}: mask must be aligned HW integer labels")
        if mask.size == 0 or mask.min() < 0 or mask.max() >= self.classes:
            raise ValueError(f"{record.identifier}: labels outside declared class range")
        return (
            torch.from_numpy(image.transpose(2, 0, 1).astype(np.float32) / 255),
            torch.from_numpy(mask.astype(np.int64)),
        )


def manifest_summary(records: tuple[SampleRecord, ...]) -> dict:
    return {
        "samples": len(records),
        "scenes": len({record.scene for record in records}),
        "splits": {
            split: sum(record.split == split for record in records)
            for split in ("train", "validation", "test")
        },
    }
