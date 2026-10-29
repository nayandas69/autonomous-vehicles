"""Convex collision and ray intersection cases independent of the simulator."""

from math import pi, sqrt

import pytest

from autonomous_vehicles.simulation.geometry import Pose2, Vec2
from autonomous_vehicles.simulation.shapes import (
    Rectangle,
    point_rectangle_distance,
    ray_rectangle_distance,
)


def test_axis_aligned_rectangle_corners_and_containment():
    box = Rectangle(Pose2(2, 3), 4, 2)
    assert box.aabb() == (0, 2, 4, 4)
    assert box.contains(Vec2(0, 2))
    assert box.contains(Vec2(2, 3))
    assert not box.contains(Vec2(4.1, 3))
    assert box.contains(Vec2(4.1, 3), margin=0.2)


def test_rotated_rectangle_containment():
    box = Rectangle(Pose2(0, 0, pi / 2), 4, 2)
    assert box.contains(Vec2(0, 1.9))
    assert not box.contains(Vec2(1.1, 0))
    minimum_x, minimum_y, maximum_x, maximum_y = box.aabb()
    assert (minimum_x, minimum_y) == pytest.approx((-1, -2))
    assert (maximum_x, maximum_y) == pytest.approx((1, 2))


def test_touching_is_collision_and_overlap_is_symmetric():
    first = Rectangle(Pose2(), 2, 2)
    touching = Rectangle(Pose2(2, 0), 2, 2)
    separated = Rectangle(Pose2(2.01, 0), 2, 2)
    assert first.overlaps(touching)
    assert touching.overlaps(first)
    assert not first.overlaps(separated)
    assert not separated.overlaps(first)


def test_rotated_collision_uses_separating_axes():
    first = Rectangle(Pose2(), 4, 1)
    crossing = Rectangle(Pose2(0, 0, pi / 2), 4, 1)
    distant = Rectangle(Pose2(8, 8, pi / 4), 4, 1)
    assert first.overlaps(crossing)
    assert not first.overlaps(distant)


def test_distance_is_zero_inside_and_euclidean_at_corner():
    box = Rectangle(Pose2(), 2, 2)
    assert point_rectangle_distance(Vec2(), box) == 0
    assert point_rectangle_distance(Vec2(2, 1), box) == 1
    assert point_rectangle_distance(Vec2(2, 2), box) == pytest.approx(sqrt(2))


def test_ray_intersection_normalizes_direction():
    box = Rectangle(Pose2(10, 0), 2, 4)
    assert ray_rectangle_distance(Vec2(0, 0), Vec2(2, 0), box) == pytest.approx(9)
    assert ray_rectangle_distance(Vec2(10, 0), Vec2(1, 0), box) == 0
    assert ray_rectangle_distance(Vec2(0, 3), Vec2(1, 0), box) is None
    assert ray_rectangle_distance(Vec2(0, 0), Vec2(-1, 0), box) is None


def test_ray_hits_rotated_rectangle():
    box = Rectangle(Pose2(5, 0, pi / 2), 4, 2)
    assert ray_rectangle_distance(Vec2(0, 0), Vec2(1, 0), box) == pytest.approx(4)


def test_expansion_contains_original_corners():
    original = Rectangle(Pose2(2, 4, 0.3), 4, 2)
    expanded = original.expanded(0.5)
    assert expanded.length == 5
    assert expanded.width == 3
    assert all(expanded.contains(corner) for corner in original.corners)
    with pytest.raises(ValueError):
        original.expanded(-1)
    with pytest.raises(ValueError):
        original.contains(Vec2(0, 0), -1)


@pytest.mark.parametrize("length,width", [(0, 1), (1, -1), (float("nan"), 2)])
def test_rectangle_rejects_invalid_dimensions(length, width):
    with pytest.raises(ValueError):
        Rectangle(Pose2(), length, width)
