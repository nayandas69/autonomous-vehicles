"""Tracking lifecycle and evaluation identities across independent frames."""

import pytest

from autonomous_vehicles.evaluation.tracking import AnnotatedBox, TrackingEvaluator
from autonomous_vehicles.perception.boxes import Box, Detection
from autonomous_vehicles.perception.sequences import MovingBox, moving_sequence
from autonomous_vehicles.perception.tracking import IoUTracker, Track, TrackerConfig


def test_track_confirmation_then_deletion_after_misses():
    tracker = IoUTracker(TrackerConfig(confirmation_hits=2, maximum_misses=1))
    detection = Detection(Box(0, 0, 10, 10), 0.9)
    first = tracker.update((detection,))[0]
    assert not first.confirmed and first.hits == 1
    second = tracker.update((detection,))[0]
    assert second.confirmed and second.identifier == first.identifier
    missed = tracker.update(())[0]
    assert missed.misses == 1
    assert tracker.update(()) == ()


def test_constant_motion_preserves_identity_and_velocity():
    tracker = IoUTracker(TrackerConfig(velocity_smoothing=1))
    identities = []
    for x in (0, 1, 2, 3):
        track = tracker.update((Detection(Box(x, 0, x + 10, 10), 0.9),), dt=0.5)[0]
        identities.append(track.identifier)
    assert len(set(identities)) == 1
    assert track.vx == pytest.approx(2)
    assert track.vy == 0


def test_category_change_creates_new_track():
    tracker = IoUTracker()
    box = Box(0, 0, 10, 10)
    first = tracker.update((Detection(box, 0.9, 1),))[0]
    tracks = tracker.update((Detection(box, 0.9, 2),))
    assert len(tracks) == 2
    assert tracks[0].identifier == first.identifier
    assert tracks[1].identifier != first.identifier


def test_tracker_reset_restarts_identity_namespace():
    tracker = IoUTracker()
    tracker.update((Detection(Box(0, 0, 10, 10), 0.9),))
    tracker.reset()
    assert tracker.tracks == ()
    assert tracker.next_identifier == 1
    with pytest.raises(ValueError):
        tracker.update((), dt=0)


def test_greedy_matching_assigns_each_track_once():
    tracker = IoUTracker()
    detection = Detection(Box(0, 0, 10, 10), 0.9)
    tracker.update((detection,))
    tracks = tracker.update((detection, detection))
    assert len(tracks) == 2
    assert len({track.identifier for track in tracks}) == 2


def test_annotated_sequence_has_repeatable_noise_and_aligned_boxes():
    objects = (MovingBox("one", Box(2, 3, 12, 13), vx=1),)
    first = moving_sequence(objects, frames=5, noise=0.1, seed=7)
    second = moving_sequence(objects, frames=5, noise=0.1, seed=7)
    assert first[2].targets == second[2].targets
    assert first[2].detections == second[2].detections
    assert first[2].targets[0].box == Box(4, 3, 14, 13)
    assert first[2].image.shape == (64, 128, 3)


def test_perfect_sequence_has_zero_identity_switches():
    sequence = moving_sequence((MovingBox("one", Box(5, 5, 15, 15)),), frames=10)
    tracker = IoUTracker()
    evaluator = TrackingEvaluator()
    for frame in sequence:
        tracks = tracker.update(frame.detections)
        evaluator.update(tracks, frame.targets)
    result = evaluator.summary()
    assert result.frames == 10
    assert result.matches == 10
    assert result.identity_switches == 0
    assert result.misses == result.false_positives == 0
    assert result.mean_matched_iou == 1
    assert result.mota_like == 1


def test_evaluator_counts_identity_switch_and_duplicate_target_error():
    box = Box(0, 0, 10, 10)
    target = AnnotatedBox("truth", box)
    evaluator = TrackingEvaluator()
    evaluator.update((Track(1, box, 2, 0.9),), (target,))
    evaluator.update((Track(2, box, 2, 0.9),), (target,))
    result = evaluator.summary()
    assert result.identity_switches == 1
    assert result.mota_like == 0.5
    with pytest.raises(ValueError):
        evaluator.update((), (target, target))
    evaluator.reset()
    assert evaluator.summary().mota_like is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"minimum_iou": -1},
        {"maximum_misses": -1},
        {"confirmation_hits": 0},
        {"velocity_smoothing": 2},
    ],
)
def test_invalid_tracker_config(kwargs):
    with pytest.raises(ValueError):
        TrackerConfig(**kwargs)
