"""Simple constant-velocity, greedy IoU tracking with explicit lifecycle.

IDs are local to a tracker instance. Association is deliberately greedy rather
than Hungarian; no appearance embeddings or Deep SORT code are used.
"""

from dataclasses import dataclass

from autonomous_vehicles.perception.boxes import Box, Detection
from autonomous_vehicles.simulation.geometry import finite


@dataclass(frozen=True)
class TrackerConfig:
    minimum_iou: float = 0.1
    maximum_misses: int = 3
    confirmation_hits: int = 2
    velocity_smoothing: float = 0.5

    def __post_init__(self):
        finite(self.minimum_iou, self.velocity_smoothing)
        if not 0 <= self.minimum_iou <= 1 or not 0 <= self.velocity_smoothing <= 1:
            raise ValueError("IoU and smoothing must be in [0, 1]")
        if self.maximum_misses < 0 or self.confirmation_hits < 1:
            raise ValueError("Invalid lifecycle thresholds")


@dataclass(frozen=True)
class Track:
    identifier: int
    box: Box
    category: int
    score: float
    hits: int = 1
    misses: int = 0
    vx: float = 0.0
    vy: float = 0.0
    confirmed: bool = False


class IoUTracker:
    def __init__(self, config: TrackerConfig | None = None):
        self.config = TrackerConfig() if config is None else config
        self.reset()

    def reset(self):
        self.tracks: tuple[Track, ...] = ()
        self.next_identifier = 1

    def update(self, detections: tuple[Detection, ...], dt: float = 1.0) -> tuple[Track, ...]:
        finite(dt)
        if dt <= 0:
            raise ValueError("Frame interval must be positive")
        predicted = [track.box.translated(track.vx * dt, track.vy * dt) for track in self.tracks]
        pairs = []
        for index, track in enumerate(self.tracks):
            for detection_index, detection in enumerate(detections):
                if track.category == detection.category:
                    overlap = predicted[index].iou(detection.box)
                    if overlap >= self.config.minimum_iou:
                        pairs.append((-overlap, index, detection_index))
        used_tracks, used_detections, updated = set(), set(), []
        for _, index, detection_index in sorted(pairs):
            if index in used_tracks or detection_index in used_detections:
                continue
            track, detection = self.tracks[index], detections[detection_index]
            old_x, old_y = track.box.center
            new_x, new_y = detection.box.center
            # If a track was missed, its stored box is already propagated to this frame's origin.
            alpha = self.config.velocity_smoothing
            vx = alpha * (new_x - old_x) / dt + (1 - alpha) * track.vx
            vy = alpha * (new_y - old_y) / dt + (1 - alpha) * track.vy
            hits = track.hits + 1
            updated.append(
                Track(
                    track.identifier,
                    detection.box,
                    detection.category,
                    detection.score,
                    hits,
                    0,
                    vx,
                    vy,
                    hits >= self.config.confirmation_hits,
                )
            )
            used_tracks.add(index)
            used_detections.add(detection_index)
        for index, track in enumerate(self.tracks):
            if index not in used_tracks and track.misses < self.config.maximum_misses:
                updated.append(
                    Track(
                        track.identifier,
                        predicted[index],
                        track.category,
                        track.score,
                        track.hits,
                        track.misses + 1,
                        track.vx,
                        track.vy,
                        track.confirmed,
                    )
                )
        for index, detection in enumerate(detections):
            if index not in used_detections:
                updated.append(
                    Track(
                        self.next_identifier,
                        detection.box,
                        detection.category,
                        detection.score,
                        confirmed=self.config.confirmation_hits == 1,
                    )
                )
                self.next_identifier += 1
        self.tracks = tuple(sorted(updated, key=lambda track: track.identifier))
        return self.tracks
