import json

import pytest
import torch
from torch.utils.data import DataLoader

from autonomous_vehicles.data import RoadScenes
from autonomous_vehicles.engine import TrainConfig, load_model, train
from autonomous_vehicles.metrics import evaluate, segmentation_metrics


def test_scene_contract_and_split_independence():
    image, mask = RoadScenes()[0]
    assert image.shape == (3, 64, 64) and image.dtype == torch.float32
    assert mask.shape == (64, 64) and mask.dtype == torch.int64
    assert 0 <= image.min() <= image.max() <= 1
    assert set(mask.unique().tolist()) == {0, 1, 2}
    assert torch.equal(image, RoadScenes()[0][0])
    assert not torch.equal(image, RoadScenes(split="test")[0][0])
    with pytest.raises(IndexError):
        RoadScenes()[-1]


def test_metrics_hand_computed():
    result = segmentation_metrics(torch.tensor([[2, 1, 0], [0, 1, 0], [0, 0, 0]]))
    assert result["pixel_accuracy"] == 0.75
    assert result["class_iou"] == [2 / 3, 0.5, None]
    assert result["mean_iou"] == pytest.approx(7 / 12)


@pytest.mark.parametrize(
    "values", [{"epochs": 0}, {"size": 16}, {"seed": -1}, {"learning_rate": float("nan")}]
)
def test_invalid_configuration(values):
    with pytest.raises(ValueError):
        TrainConfig(**values)


def test_training_roundtrip_reproducibility_and_no_overwrite(tmp_path):
    config = TrainConfig(epochs=2, samples=8, size=32, batch_size=4)
    first = train(config, tmp_path / "a")
    second = train(config, tmp_path / "b")
    assert first == second
    assert first["history"][-1]["train_loss"] < first["history"][0]["train_loss"]
    model, restored = load_model(tmp_path / "a/model.pt")
    assert restored == config
    result = evaluate(model, DataLoader(RoadScenes(4, 32, split="test"), batch_size=2))
    assert 0 <= result["mean_iou"] <= 1
    assert json.loads((tmp_path / "a/training.json").read_text())["config"]["seed"] == 42
    with pytest.raises(FileExistsError):
        train(config, tmp_path / "a")
