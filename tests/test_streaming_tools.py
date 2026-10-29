"""Streaming data, confusion aggregation and artifact export boundary tests."""

import numpy as np
import pytest
import torch

from autonomous_vehicles.evaluation.profiling import profile_inference
from autonomous_vehicles.evaluation.segmentation import ConfusionMeter
from autonomous_vehicles.experiments.streams import ObservationBuffer, TimedObservation, synchronize
from autonomous_vehicles.model import RoadNet
from autonomous_vehicles.perception.dataset import ManifestDataset
from autonomous_vehicles.perception.export import export_scenes, verify_export


def test_confusion_meter_ignores_only_declared_target_labels():
    meter = ConfusionMeter(3, ignore_index=255)
    prediction = torch.tensor([[0, 1], [1, 99]])
    target = torch.tensor([[0, 1], [2, 255]])
    meter.update(prediction, target)
    result = meter.summary()
    assert result.evaluated_pixels == 3
    assert result.ignored_pixels == 1
    assert result.pixel_accuracy == pytest.approx(2 / 3)
    assert result.class_iou == (1, 0.5, 0)
    assert result.mean_iou == 0.5


def test_confusion_merge_matches_complete_batch_and_reset():
    first, second, combined = ConfusionMeter(), ConfusionMeter(), ConfusionMeter()
    first.update(torch.tensor([0, 1]), torch.tensor([0, 2]))
    second.update(torch.tensor([2, 1]), torch.tensor([2, 1]))
    combined.update(torch.tensor([0, 1, 2, 1]), torch.tensor([0, 2, 2, 1]))
    first.merge(second)
    assert first.summary() == combined.summary()
    first.reset()
    assert first.summary().mean_iou is None
    assert first.summary().pixel_accuracy is None
    with pytest.raises(ValueError):
        first.merge(ConfusionMeter(4))


def test_bad_confusion_update_is_transactional():
    meter = ConfusionMeter()
    meter.update(torch.tensor([0]), torch.tensor([0]))
    original = meter.matrix.clone()
    with pytest.raises(ValueError):
        meter.update(torch.tensor([9]), torch.tensor([0]))
    torch.testing.assert_close(meter.matrix, original)


def test_buffer_never_returns_future_or_stale_samples():
    buffer = ObservationBuffer[str](capacity=2)
    buffer.append(TimedObservation(1, "a"))
    buffer.append(TimedObservation(2, "b"))
    assert buffer.latest_before(0) is None
    assert buffer.latest_before(1.5).value == "a"
    assert buffer.latest_before(3, maximum_age=0.5) is None
    assert buffer.latest_before(2).value == "b"
    buffer.append(TimedObservation(3, "c"))
    assert len(buffer) == 2
    assert buffer.latest_before(1) is None
    assert [item.value for item in buffer.between(2, 3)] == ["b", "c"]


def test_buffer_rejects_duplicate_and_out_of_order_timestamps():
    buffer = ObservationBuffer()
    buffer.append(TimedObservation(2, 1))
    with pytest.raises(ValueError):
        buffer.append(TimedObservation(2, 2))
    with pytest.raises(ValueError):
        buffer.append(TimedObservation(1, 2))
    assert len(buffer) == 1
    buffer.clear()
    assert len(buffer) == 0


def test_synchronization_reports_missing_sensor_without_future_leakage():
    camera, lidar = ObservationBuffer(), ObservationBuffer()
    camera.append(TimedObservation(1, "camera-1"))
    lidar.append(TimedObservation(1.2, "lidar-future"))
    result = synchronize({"camera": camera, "lidar": lidar}, 1.1, 0.5)
    assert not result["ready"]
    assert result["values"] == {"camera": "camera-1", "lidar": None}
    assert result["ages"]["camera"] == pytest.approx(0.1)


def test_synthetic_export_roundtrip_and_hash_tampering(tmp_path):
    root = tmp_path / "dataset"
    manifest = export_scenes(root, samples_per_split=2, size=32)
    dataset = ManifestDataset(manifest, "train")
    assert len(dataset) == 2
    image, mask = dataset[0]
    assert image.shape == (3, 32, 32)
    assert mask.shape == (32, 32)
    assert verify_export(root)["verified"]
    path = root / "train/train-000000-mask.npy"
    labels = np.load(path)
    labels[0, 0] = (labels[0, 0] + 1) % 3
    np.save(path, labels)
    report = verify_export(root)
    assert not report["verified"]
    assert report["changed_files"] == ["train/train-000000-mask.npy"]
    with pytest.raises(FileExistsError):
        export_scenes(root)


def test_cpu_profiler_preserves_mode_and_reports_shape():
    model = RoadNet().train()
    images = torch.zeros(2, 3, 32, 32)
    report = profile_inference(model, images, warmup=1, iterations=3)
    assert model.training
    assert report.batch_size == 2
    assert report.input_shape == (2, 3, 32, 32)
    assert report.mean_seconds > 0
    assert report.images_per_second > 0
    assert report.parameters == sum(parameter.numel() for parameter in model.parameters())
