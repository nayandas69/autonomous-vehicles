"""Sequence tracking metrics using explicit annotated identity matches.

Matching follows greedy IoU association. The summary reports identity switches
and localization overlap; it does not claim compatibility with a named benchmark.
"""

from dataclasses import dataclass

from autonomous_vehicles.evaluation.detection import match_detections
from autonomous_vehicles.perception.boxes import Box, Detection
from autonomous_vehicles.perception.tracking import Track


@dataclass(frozen=True)
class AnnotatedBox:
    identifier: str
    box: Box
    category: int = 2


@dataclass(frozen=True)
class TrackingSummary:
    frames: int
    ground_truth_instances: int
    matches: int
    false_positives: int
    misses: int
    identity_switches: int
    mean_matched_iou: float | None

    @property
    def mota_like(self) -> float | None:
        if self.ground_truth_instances == 0:
            return None
        return (
            1
            - (self.false_positives + self.misses + self.identity_switches)
            / self.ground_truth_instances
        )


class TrackingEvaluator:
    def __init__(self, iou_threshold: float = 0.5):
        if not 0 <= iou_threshold <= 1:
            raise ValueError("IoU threshold must lie in [0, 1]")
        self.iou_threshold = iou_threshold
        self.reset()

    def reset(self):
        self.frames = 0
        self.instances = 0
        self.matches = 0
        self.false_positives = 0
        self.misses = 0
        self.switches = 0
        self.overlaps = []
        self.previous_ids = {}

    def update(self, tracks: tuple[Track, ...], targets: tuple[AnnotatedBox, ...]):
        if len({target.identifier for target in targets}) != len(targets):
            raise ValueError("Ground-truth IDs must be unique within a frame")
        if len({track.identifier for track in tracks}) != len(tracks):
            raise ValueError("Track IDs must be unique within a frame")
        detections = tuple(Detection(track.box, track.score, track.category) for track in tracks)
        labels = tuple((target.box, target.category) for target in targets)
        result = match_detections(detections, labels, self.iou_threshold)
        self.frames += 1
        self.instances += len(targets)
        self.matches += result.true_positives
        self.false_positives += result.false_positives
        self.misses += result.false_negatives
        for track_index, target_index in result.matched_pairs:
            track, target = tracks[track_index], targets[target_index]
            previous = self.previous_ids.get(target.identifier)
            if previous is not None and previous != track.identifier:
                self.switches += 1
            self.previous_ids[target.identifier] = track.identifier
            self.overlaps.append(track.box.iou(target.box))
        return result

    def summary(self) -> TrackingSummary:
        average = sum(self.overlaps) / len(self.overlaps) if self.overlaps else None
        return TrackingSummary(
            self.frames,
            self.instances,
            self.matches,
            self.false_positives,
            self.misses,
            self.switches,
            average,
        )
