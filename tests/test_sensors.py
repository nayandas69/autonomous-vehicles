"""Seeded sensing, field of view and inverse sensor model behavior."""

from math import pi

import numpy as np
import pytest

from autonomous_vehicles.simulation.actors import ActorState
from autonomous_vehicles.simulation.geometry import Pose2, Vec2
from autonomous_vehicles.simulation.occupancy import GridSpec, OccupancyGrid
from autonomous_vehicles.simulation.sensors import RangeSensor, RangeSensorConfig, visible_actors


def test_central_ray_hits_front_face():
    actor = ActorState("front", Pose2(10, 0), 0, length=2, width=2)
    config = RangeSensorConfig(beams=3, field_of_view=pi / 2, noise_std=0)
    scan = RangeSensor(config).scan(Pose2(), (actor,))
    assert scan.ranges[1] == pytest.approx(9)
    assert scan.hits[1]
    assert not scan.hits[0]
    assert not scan.hits[2]
    np.testing.assert_allclose(scan.points()[1], [9, 0], atol=1e-12)


def test_empty_scene_and_full_dropout_are_unknown_returns():
    config = RangeSensorConfig(beams=5, maximum_range=20, dropout=1)
    scan = RangeSensor(config).scan(Pose2(), (ActorState("front", Pose2(10, 0), 0),))
    assert not scan.hits.any()
    np.testing.assert_array_equal(scan.ranges, np.full(5, 20))
    empty = RangeSensor(config).scan(Pose2(), ())
    assert not empty.hits.any()


def test_noise_is_repeatable_for_same_seed():
    actor = ActorState("front", Pose2(10, 0), 0, width=10)
    config = RangeSensorConfig(beams=9, noise_std=0.2)
    first = RangeSensor(config, seed=7).scan(Pose2(), (actor,))
    second = RangeSensor(config, seed=7).scan(Pose2(), (actor,))
    np.testing.assert_array_equal(first.ranges, second.ranges)
    assert np.all(first.ranges >= 0)
    assert np.all(first.ranges <= config.maximum_range)


def test_scan_world_coordinates_respect_pose():
    sensor = RangeSensor(RangeSensorConfig(beams=3, noise_std=0))
    pose = Pose2(2, 3, pi / 2)
    actor = ActorState("north", Pose2(2, 13), 0, length=2, width=2)
    scan = sensor.scan(pose, (actor,))
    np.testing.assert_allclose(scan.points()[1], [2, 12], atol=1e-12)
    np.testing.assert_allclose(scan.points(world=False)[1], [9, 0], atol=1e-12)


def test_visibility_is_sorted_and_excludes_rear_actor():
    actors = (
        ActorState("far", Pose2(20, 0), 0),
        ActorState("rear", Pose2(-10, 0), 0),
        ActorState("near", Pose2(5, 0), 0),
    )
    visible = visible_actors(Pose2(), actors)
    assert [actor.identifier for actor in visible] == ["near", "far"]
    assert visible[0].distance == 5
    assert visible[0].bearing == 0


def test_range_integration_marks_free_and_hit_cells():
    spec = GridSpec(width=30, height=20, resolution=1, origin=Vec2(-1, -10))
    grid = OccupancyGrid(spec)
    sensor = RangeSensor(RangeSensorConfig(beams=3, noise_std=0))
    scan = sensor.scan(Pose2(), (ActorState("front", Pose2(10, 0), 0, 2, 2),))
    grid.integrate(scan)
    assert grid.probabilities[spec.cell(Vec2(4, 0))] < 0.5
    assert grid.probabilities[spec.cell(Vec2(9, 0))] > 0.5
    assert grid.probabilities[spec.cell(Vec2(15, 0))] == 0.5


def test_dropout_does_not_clear_occupancy():
    grid = OccupancyGrid()
    original = grid.log_odds.copy()
    scan = RangeSensor(RangeSensorConfig(dropout=1)).scan(Pose2(), ())
    grid.integrate(scan)
    np.testing.assert_array_equal(grid.log_odds, original)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"beams": 1},
        {"field_of_view": 0},
        {"maximum_range": -1},
        {"noise_std": -0.1},
        {"dropout": 1.1},
    ],
)
def test_invalid_range_config(kwargs):
    with pytest.raises(ValueError):
        RangeSensorConfig(**kwargs)
