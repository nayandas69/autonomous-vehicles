"""Local dataset isolation, path confinement, labels and split leakage checks."""

import json

import numpy as np
import pytest

from autonomous_vehicles.perception.dataset import (
    ManifestDataset,
    confined_path,
    load_manifest,
    manifest_summary,
)


def make_manifest(root, records=None):
    image = np.zeros((8, 10, 3), dtype=np.uint8)
    image[..., 0] = 255
    mask = np.zeros((8, 10), dtype=np.uint8)
    mask[2:4, 3:5] = 2
    np.save(root / "image.npy", image)
    np.save(root / "mask.npy", mask)
    if records is None:
        records = [
            {
                "identifier": "one",
                "scene": "scene-a",
                "split": "train",
                "image": "image.npy",
                "mask": "mask.npy",
            }
        ]
    path = root / "manifest.json"
    path.write_text(json.dumps({"format_version": 1, "samples": records}))
    return path


def test_manifest_dataset_converts_hwc_uint8_and_integer_labels(tmp_path):
    path = make_manifest(tmp_path)
    dataset = ManifestDataset(path, "train")
    assert len(dataset) == 1
    image, mask = dataset[0]
    assert image.shape == (3, 8, 10)
    assert image[0].min() == image[0].max() == 1
    assert mask[2, 3] == 2
    summary = manifest_summary(load_manifest(path))
    assert summary == {
        "samples": 1,
        "scenes": 1,
        "splits": {"train": 1, "validation": 0, "test": 0},
    }


def test_manifest_rejects_scene_leakage(tmp_path):
    path = make_manifest(tmp_path)
    np.save(tmp_path / "other.npy", np.zeros((8, 10, 3), dtype=np.uint8))
    payload = json.loads(path.read_text())
    second = {**payload["samples"][0], "identifier": "two", "split": "test", "image": "other.npy"}
    payload["samples"].append(second)
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="leaks"):
        load_manifest(path)


def test_manifest_rejects_duplicate_id_and_reused_image(tmp_path):
    path = make_manifest(tmp_path)
    payload = json.loads(path.read_text())
    payload["samples"].append(dict(payload["samples"][0]))
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Duplicate"):
        load_manifest(path)
    payload["samples"][1]["identifier"] = "two"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="multiple"):
        load_manifest(path)


def test_dataset_paths_cannot_escape_via_parent_or_symlink(tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        confined_path(tmp_path, "../secret.npy")
    with pytest.raises(ValueError, match="relative"):
        confined_path(tmp_path, "/tmp/file.npy")
    (tmp_path / "escape").symlink_to(tmp_path.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        confined_path(tmp_path, "escape/outside.npy")


def test_manifest_missing_split_and_bad_labels_are_errors(tmp_path):
    path = make_manifest(tmp_path)
    with pytest.raises(ValueError, match="No samples"):
        ManifestDataset(path, "test")
    np.save(tmp_path / "mask.npy", np.full((8, 10), 3, dtype=np.uint8))
    with pytest.raises(ValueError, match="range"):
        ManifestDataset(path, "train")[0]


def test_manifest_rejects_wrong_image_contract(tmp_path):
    path = make_manifest(tmp_path)
    np.save(tmp_path / "image.npy", np.zeros((3, 8, 10), dtype=np.float32))
    with pytest.raises(ValueError, match="HWC uint8"):
        ManifestDataset(path, "train")[0]
    path.write_text(json.dumps({"format_version": 9, "samples": []}))
    with pytest.raises(ValueError, match="format"):
        load_manifest(path)
