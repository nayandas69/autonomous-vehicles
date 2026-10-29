"""Class-aware greedy detection matching and precision-recall summaries."""

from dataclasses import dataclass

import numpy as np

from autonomous_vehicles.perception.boxes import Box, Detection


@dataclass(frozen=True)
class MatchSummary:
    true_positives: int
    false_positives: int
    false_negatives: int
    matched_pairs: tuple[tuple[int, int], ...]

    @property
    def precision(self) -> float:
        total = self.true_positives + self.false_positives
        return self.true_positives / total if total else 0.0

    @property
    def recall(self) -> float:
        total = self.true_positives + self.false_negatives
        return self.true_positives / total if total else 0.0

    @property
    def f1(self) -> float:
        total = self.precision + self.recall
        return 2 * self.precision * self.recall / total if total else 0.0


def match_detections(
    detections: tuple[Detection, ...],
    targets: tuple[tuple[Box, int], ...],
    iou_threshold: float = 0.5,
) -> MatchSummary:
    if not 0 <= iou_threshold <= 1:
        raise ValueError("IoU threshold must lie in [0, 1]")
    matched, pairs = set(), []
    order = sorted(range(len(detections)), key=lambda index: (-detections[index].score, index))
    for detection_index in order:
        detection = detections[detection_index]
        available = [
            (detection.box.iou(box), index)
            for index, (box, category) in enumerate(targets)
            if index not in matched and category == detection.category
        ]
        if not available:
            continue
        overlap, target_index = max(available, key=lambda pair: (pair[0], -pair[1]))
        if overlap >= iou_threshold:
            matched.add(target_index)
            pairs.append((detection_index, target_index))
    return MatchSummary(
        len(pairs), len(detections) - len(pairs), len(targets) - len(pairs), tuple(pairs)
    )


@dataclass(frozen=True)
class PrecisionRecall:
    thresholds: np.ndarray
    precision: np.ndarray
    recall: np.ndarray


def precision_recall_curve(
    detections: tuple[Detection, ...],
    targets: tuple[tuple[Box, int], ...],
    thresholds: tuple[float, ...] = (0, 0.25, 0.5, 0.75, 1),
) -> PrecisionRecall:
    if not thresholds or any(not 0 <= value <= 1 for value in thresholds):
        raise ValueError("Need thresholds in [0, 1]")
    results = [
        match_detections(tuple(d for d in detections if d.score >= threshold), targets)
        for threshold in thresholds
    ]
    return PrecisionRecall(
        np.array(thresholds),
        np.array([result.precision for result in results]),
        np.array([result.recall for result in results]),
    )
