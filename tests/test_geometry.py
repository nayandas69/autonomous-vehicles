"""Independent geometric identities and hand-computed reference cases."""

from math import pi

import numpy as np
import pytest

from autonomous_vehicles.simulation.geometry import (
    Pose2,
    Vec2,
    angle_difference,
    clamp,
    finite,
    heading,
    interpolate_pose,
    polyline_lengths,
    project_segment,
    wrap_angle,
)


def test_rotation_and_inverse_preserve_distance():
    vector = Vec2(3, 4)
    rotated = vector.rotate(pi / 2)
    assert rotated.x == pytest.approx(-4)
    assert rotated.y == pytest.approx(3)
    assert rotated.norm == pytest.approx(5)
    restored = rotated.rotate(-pi / 2)
    assert restored.x == pytest.approx(vector.x)
    assert restored.y == pytest.approx(vector.y)


def test_vector_operations_use_metric_coordinates():
    first, second = Vec2(2, 3), Vec2(-1, 4)
    assert first + second == Vec2(1, 7)
    assert first - second == Vec2(3, -1)
    assert first * 2 == Vec2(4, 6)
    assert first.dot(second) == 10
    assert first.cross(second) == 11
    assert first.normalized().norm == pytest.approx(1)
    np.testing.assert_array_equal(first.array(), [2, 3])


def test_normalizing_zero_is_an_error():
    with pytest.raises(ValueError, match="zero"):
        Vec2(0, 0).normalized()


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_geometry_rejects_nonfinite_values(invalid):
    for operation in (
        lambda: finite(invalid),
        lambda: Vec2(invalid, 0),
        lambda: Pose2(0, 0, invalid),
        lambda: wrap_angle(invalid),
        lambda: clamp(invalid, 0, 1),
    ):
        with pytest.raises(ValueError):
            operation()


def test_pose_roundtrip_and_composition():
    pose = Pose2(10, -3, pi / 3)
    point = Vec2(5, -2)
    world = pose.local_to_world(point)
    restored = pose.world_to_local(world)
    np.testing.assert_allclose(restored.array(), point.array(), atol=1e-12)
    identity = pose.compose(pose.inverse())
    assert identity.x == pytest.approx(0, abs=1e-12)
    assert identity.y == pytest.approx(0, abs=1e-12)
    assert identity.yaw == pytest.approx(0, abs=1e-12)


def test_heading_wrap_uses_shortest_rotation():
    assert angle_difference(-pi + 0.1, pi - 0.1) == pytest.approx(0.2)
    midpoint = interpolate_pose(Pose2(0, 0, pi - 0.1), Pose2(2, 4, -pi + 0.1), 0.5)
    assert midpoint.x == 1
    assert midpoint.y == 2
    assert abs(midpoint.yaw) == pytest.approx(pi)
    assert interpolate_pose(Pose2(), Pose2(2, 4), 2).position == Vec2(2, 4)


def test_projection_on_segment_and_endpoints():
    result = project_segment(Vec2(3, 4), Vec2(0, 0), Vec2(10, 0))
    assert result.point == Vec2(3, 0)
    assert result.fraction == pytest.approx(0.3)
    assert result.distance == 4
    assert result.signed_distance == 4
    result = project_segment(Vec2(-2, -1), Vec2(0, 0), Vec2(10, 0))
    assert result.fraction == 0
    assert result.signed_distance < 0
    degenerate = project_segment(Vec2(3, 4), Vec2(0, 0), Vec2(0, 0))
    assert degenerate.distance == 5


def test_polyline_lengths_and_heading():
    lengths = polyline_lengths((Vec2(0, 0), Vec2(3, 4), Vec2(3, 8)))
    np.testing.assert_array_equal(lengths, [0, 5, 9])
    assert heading(Vec2(0, 0), Vec2(0, 1)) == pytest.approx(pi / 2)
    with pytest.raises(ValueError):
        heading(Vec2(0, 0), Vec2(0, 0))
    with pytest.raises(ValueError):
        polyline_lengths((Vec2(0, 0), Vec2(0, 0)))
    with pytest.raises(ValueError):
        clamp(0, 2, 1)
