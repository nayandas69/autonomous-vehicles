"""Road projection, bounds, sampling, and curvature behavior."""

from math import pi

import numpy as np
import pytest

from autonomous_vehicles.simulation.geometry import Vec2
from autonomous_vehicles.simulation.roads import Road, circular_road, curved_road, straight_road


def test_straight_road_arc_length_and_lateral_sign():
    road = straight_road(100)
    projection = road.project(Vec2(20, 2))
    assert projection.s == pytest.approx(20)
    assert projection.lateral == pytest.approx(2)
    assert projection.distance == pytest.approx(2)
    assert projection.pose.yaw == 0
    assert road.pose_at(20, -2).position == Vec2(20, -2)
    assert road.length == 100


def test_projection_clips_to_finite_road():
    road = straight_road(10)
    assert road.project(Vec2(-3, 0)).s == 0
    assert road.project(Vec2(13, 0)).s == 10
    assert road.pose_at(-5).x == 0
    assert road.pose_at(15).x == 10
    assert not road.contains(Vec2(-10, 0))


def test_road_width_and_margin():
    road = straight_road(width=6)
    assert road.contains(Vec2(10, 3))
    assert not road.contains(Vec2(10, 3.1))
    assert not road.contains(Vec2(10, 2.5), margin=1)
    with pytest.raises(ValueError):
        road.contains(Vec2(10, 0), margin=3)


def test_arc_length_sampling_includes_both_endpoints():
    road = straight_road(10)
    samples = road.samples(spacing=3)
    assert len(samples) == 5
    assert samples[0].x == 0
    assert samples[-1].x == 10
    assert max(b.x - a.x for a, b in zip(samples[:-1], samples[1:], strict=True)) <= 3


def test_curved_road_projection_roundtrip():
    road = curved_road(length=100, amplitude=5)
    for s in np.linspace(5, road.length - 5, 10):
        pose = road.pose_at(float(s), 0.2)
        projected = road.project(pose.position)
        assert projected.s == pytest.approx(s, abs=0.03)
        assert projected.lateral == pytest.approx(0.2, abs=0.01)


def test_circular_road_curvature_is_inverse_radius():
    road = circular_road(radius=40, samples=401)
    assert road.length == pytest.approx(40 * pi / 2, rel=1e-4)
    assert road.curvature_at(road.length / 2, 4) == pytest.approx(1 / 40, rel=0.03)
    assert road.pose_at(road.length).x == pytest.approx(40)
    assert road.pose_at(road.length).y == pytest.approx(40)


def test_straight_curvature_is_zero():
    road = straight_road()
    assert road.curvature_at(20) == 0
    assert road.curvature_at(0) == 0


@pytest.mark.parametrize(
    "factory",
    [
        lambda: straight_road(0),
        lambda: curved_road(wavelength=0),
        lambda: circular_road(radius=-1),
        lambda: Road((Vec2(0, 0), Vec2(0, 0))),
        lambda: straight_road(width=0),
    ],
)
def test_invalid_road_geometry(factory):
    with pytest.raises(ValueError):
        factory()


def test_road_arrays_are_read_only():
    road = straight_road()
    with pytest.raises(ValueError):
        road.lengths[0] = 5
    with pytest.raises(ValueError):
        road.samples(0)
