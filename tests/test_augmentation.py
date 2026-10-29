"""Augmentation coordinate alignment, isolation, repeatability and corruption extremes."""

import pytest
import torch

from autonomous_vehicles.data import RoadScenes
from autonomous_vehicles.evaluation.robustness import CorruptedDataset, evaluate_corruptions
from autonomous_vehicles.model import RoadNet
from autonomous_vehicles.perception.augmentation import (
    AugmentationConfig,
    Augmenter,
    image_corruption,
    validate_sample,
)
from autonomous_vehicles.perception.inference import Predictor


def test_forced_flip_preserves_label_alignment_and_input_storage():
    image, mask = RoadScenes(size=32)[0]
    before_image, before_mask = image.clone(), mask.clone()
    augmenter = Augmenter(AugmentationConfig(flip_probability=1, brightness=0, contrast=0, noise=0))
    result, labels = augmenter(image, mask)
    torch.testing.assert_close(result, image.flip(-1))
    torch.testing.assert_close(labels, mask.flip(-1))
    torch.testing.assert_close(image, before_image)
    torch.testing.assert_close(mask, before_mask)


def test_seeded_augmentation_is_reproducible_and_bounded():
    image, mask = RoadScenes(size=32)[0]
    first = Augmenter(seed=7)(image, mask)
    second = Augmenter(seed=7)(image, mask)
    torch.testing.assert_close(first[0], second[0])
    torch.testing.assert_close(first[1], second[1])
    assert first[0].min() >= 0 and first[0].max() <= 1


@pytest.mark.parametrize("kind", ["darkness", "fog", "noise", "occlusion"])
def test_zero_severity_is_identity(kind):
    image, _ = RoadScenes(size=32)[0]
    torch.testing.assert_close(image_corruption(image, kind, 0), image)


def test_corruption_extremes():
    image, _ = RoadScenes(size=32)[0]
    assert image_corruption(image, "darkness", 1).sum() == 0
    assert image_corruption(image, "occlusion", 1).sum() == 0
    torch.testing.assert_close(image_corruption(image, "fog", 1), torch.full_like(image, 0.8))


def test_corruption_dataset_keeps_ground_truth_and_repeatable_samples():
    base = RoadScenes(4, size=32, split="test")
    corrupted = CorruptedDataset(base, "noise", 0.5)
    assert len(corrupted) == len(base)
    first, mask = corrupted[0]
    second, _ = corrupted[0]
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(mask, base[0][1])


def test_robustness_result_has_one_row_per_condition():
    torch.set_num_threads(1)
    results = evaluate_corruptions(
        RoadNet(), samples=2, size=32, severities=(0, 0.5), kinds=("darkness",)
    )
    assert len(results) == 2
    assert [result.severity for result in results] == [0, 0.5]
    assert all(0 <= result.mean_iou <= 1 for result in results)


def test_predictor_batched_probabilities_and_single_prediction():
    image, _ = RoadScenes(size=32)[0]
    predictor = Predictor(RoadNet(), flip_average=True)
    probabilities = predictor.probabilities(image.unsqueeze(0))
    assert probabilities.shape == (1, 3, 32, 32)
    torch.testing.assert_close(probabilities.sum(1), torch.ones(1, 32, 32))
    result = predictor.predict(image)
    assert result.mask.shape == (32, 32)
    assert result.confidence.min() >= 0 and result.confidence.max() <= 1
    assert result.elapsed_seconds >= 0


@pytest.mark.parametrize(
    "operation",
    [
        lambda: AugmentationConfig(brightness=2),
        lambda: image_corruption(torch.ones(3, 4, 4), "missing", 0.5),
        lambda: validate_sample(torch.ones(3, 4, 4) * 2, torch.zeros(4, 4, dtype=torch.int64)),
        lambda: CorruptedDataset(RoadScenes(), "missing", 0.5),
    ],
)
def test_invalid_augmentation_contract(operation):
    with pytest.raises(ValueError):
        operation()
