"""Image geometry, suppression and connected foreground components."""

import numpy as np
import pytest

from autonomous_vehicles.evaluation.detection import match_detections, precision_recall_curve
from autonomous_vehicles.perception.boxes import Box, Detection, pairwise_iou, suppress
from autonomous_vehicles.perception.components import connected_components, mask_detections


def test_box_area_intersection_and_half_open_edges():
    first = Box(0, 0, 4, 4)
    second = Box(2, 0, 6, 4)
    assert first.area == 16
    assert first.center == (2, 2)
    assert first.intersection(second) == 8
    assert first.iou(second) == pytest.approx(1 / 3)
    assert first.iou(Box(4, 0, 8, 4)) == 0
    assert first.iou(first) == 1


def test_box_clipping_and_normalization():
    box = Box(-2, -1, 5, 4)
    assert box.clip(4, 3) == Box(0, 0, 4, 3)
    assert Box(-4, -4, -2, -2).clip(10, 10) is None
    assert Box(0, 0, 10, 20).normalized(10, 20) == Box(0, 0, 1, 1)
    assert Box(0, 0, 2, 2).translated(3, 4) == Box(3, 4, 5, 6)


def test_suppression_is_class_aware_and_deterministic():
    box = Box(0, 0, 10, 10)
    first = Detection(box, 0.9, 1)
    duplicate = Detection(box, 0.8, 1)
    other_class = Detection(box, 0.7, 2)
    assert suppress((duplicate, first, other_class)) == (first, other_class)
    assert suppress((first, duplicate), threshold=1) == (first, duplicate)
    assert suppress(()) == ()


def test_pairwise_overlap_shape_including_empty_sides():
    box = Box(0, 0, 2, 2)
    np.testing.assert_array_equal(pairwise_iou((box,), (box,)), [[1]])
    assert pairwise_iou((), (box,)).shape == (0, 1)


def test_components_are_four_connected_and_use_pixel_edges():
    mask = np.zeros((8, 10), dtype=np.int64)
    mask[1:3, 2:5] = 2
    mask[5:7, 7:9] = 2
    mask[0, 0] = 2
    result = connected_components(mask, minimum_pixels=4)
    assert len(result) == 2
    assert result[0].box == Box(2, 1, 5, 3)
    assert result[0].pixels == 6
    assert result[0].centroid == (3.5, 2)
    diagonal = np.array([[2, 0], [0, 2]])
    assert len(connected_components(diagonal, minimum_pixels=1)) == 2


def test_mask_detections_use_foreground_confidence():
    probabilities = np.zeros((3, 6, 6), dtype=float)
    probabilities[0] = 1
    probabilities[0, 1:4, 2:5] = 0.1
    probabilities[2, 1:4, 2:5] = 0.9
    result = mask_detections(probabilities)
    assert len(result) == 1
    assert result[0].box == Box(2, 1, 5, 4)
    assert result[0].score == pytest.approx(0.9)
    assert mask_detections(probabilities, threshold=0.95) == ()


def test_detection_matching_never_matches_target_twice():
    box = Box(0, 0, 4, 4)
    detections = (Detection(box, 0.9), Detection(box, 0.8))
    result = match_detections(detections, ((box, 2),))
    assert result.true_positives == 1
    assert result.false_positives == 1
    assert result.false_negatives == 0
    assert result.precision == 0.5
    assert result.recall == 1
    assert result.f1 == pytest.approx(2 / 3)


def test_detection_matching_respects_category():
    box = Box(0, 0, 4, 4)
    result = match_detections((Detection(box, 0.9, 1),), ((box, 2),))
    assert result.true_positives == 0
    assert result.false_positives == 1
    assert result.false_negatives == 1
    empty = match_detections((), ())
    assert empty.precision == empty.recall == empty.f1 == 0


def test_threshold_curve_drops_low_confidence_duplicates():
    box = Box(0, 0, 4, 4)
    curve = precision_recall_curve((Detection(box, 0.9), Detection(box, 0.2)), ((box, 2),))
    assert curve.precision[0] == 0.5
    assert curve.precision[2] == 1
    assert curve.recall[-1] == 0


@pytest.mark.parametrize(
    "operation",
    [
        lambda: Box(1, 1, 1, 2),
        lambda: Detection(Box(0, 0, 1, 1), 2),
        lambda: connected_components(np.zeros((2, 2), dtype=float)),
        lambda: mask_detections(np.ones((3, 2, 2)) * 2),
        lambda: suppress((), -1),
    ],
)
def test_invalid_perception_geometry(operation):
    with pytest.raises(ValueError):
        operation()
