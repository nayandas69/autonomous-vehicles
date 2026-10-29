"""Export small synthetic fixtures through the same manifest contract as real adapters."""

import json
from pathlib import Path

import numpy as np

from autonomous_vehicles.data import RoadScenes
from autonomous_vehicles.experiments.artifacts import atomic_json, file_sha256
from autonomous_vehicles.perception.dataset import load_manifest


def export_scenes(root: Path, samples_per_split: int = 8, size: int = 64, seed: int = 42) -> Path:
    # Validate before reserving a destination or writing partial output.
    RoadScenes(samples_per_split, size, seed)
    root.mkdir(parents=True, exist_ok=False)
    records, hashes = [], {}
    for split in ("train", "validation", "test"):
        dataset = RoadScenes(samples_per_split, size, seed, split)
        directory = root / split
        directory.mkdir()
        for index in range(len(dataset)):
            image, mask = dataset[index]
            identifier = f"{split}-{index:06d}"
            image_path = directory / f"{identifier}-image.npy"
            mask_path = directory / f"{identifier}-mask.npy"
            rgb = np.round(image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            np.save(image_path, rgb, allow_pickle=False)
            np.save(mask_path, mask.numpy(), allow_pickle=False)
            relative_image = str(image_path.relative_to(root))
            relative_mask = str(mask_path.relative_to(root))
            records.append(
                {
                    "identifier": identifier,
                    "scene": identifier,
                    "split": split,
                    "image": relative_image,
                    "mask": relative_mask,
                }
            )
            hashes[relative_image] = file_sha256(image_path)
            hashes[relative_mask] = file_sha256(mask_path)
    manifest = root / "manifest.json"
    atomic_json(manifest, {"format_version": 1, "samples": records})
    atomic_json(
        root / "provenance.json",
        {
            "format_version": 1,
            "generator": "RoadScenes",
            "samples_per_split": samples_per_split,
            "size": size,
            "seed": seed,
            "sha256": hashes,
        },
    )
    load_manifest(manifest)
    return manifest


def verify_export(root: Path) -> dict:
    provenance = json.loads((root / "provenance.json").read_text())
    if provenance.get("format_version") != 1:
        raise ValueError("Unsupported provenance format")
    records = load_manifest(root / "manifest.json")
    from autonomous_vehicles.perception.dataset import confined_path

    failures = []
    for relative, expected in provenance["sha256"].items():
        path = confined_path(root, relative)
        if file_sha256(path) != expected:
            failures.append(relative)
    return {"samples": len(records), "verified": not failures, "changed_files": failures}
