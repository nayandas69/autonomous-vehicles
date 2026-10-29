"""Cross-component numerical invariants and boundary regressions."""

from dataclasses import replace
from math import pi

import numpy as np
import pytest
import torch

from autonomous_vehicles.control.driver import Driver
from autonomous_vehicles.evaluation.segmentation import ConfusionMeter
from autonomous_vehicles.planning.polynomial import Boundary, Quintic
from autonomous_vehicles.simulation.dynamics import Control, VehicleState, bicycle_step
from autonomous_vehicles.simulation.geometry import Pose2, Vec2
from autonomous_vehicles.simulation.scenarios import make_scenario
from autonomous_vehicles.simulation.shapes import Rectangle, ray_rectangle_distance
from autonomous_vehicles.simulation.world import run_episode


def test_rotation_equivariance_of_bicycle_motion():
    base = VehicleState(Pose2(3, -2, 0.2), speed=4)
    transform = Pose2(10, 8, 1.1)
    transformed = replace(base, pose=transform.compose(base.pose))
    control = Control(0.5, 0.2)
    expected = transform.compose(bicycle_step(base, control, 0.1).pose)
    observed = bicycle_step(transformed, control, 0.1).pose
    np.testing.assert_allclose(
        [observed.x, observed.y, observed.yaw], [expected.x, expected.y, expected.yaw], atol=1e-12
    )


def test_collision_queries_are_invariant_to_rigid_world_transform():
    transform = Pose2(20, -8, pi / 3)
    first = Rectangle(Pose2(0, 0, 0.2), 4, 2)
    second = Rectangle(Pose2(2, 1, -0.4), 4, 2)
    transformed_first = Rectangle(transform.compose(first.pose), first.length, first.width)
    transformed_second = Rectangle(transform.compose(second.pose), second.length, second.width)
    assert first.overlaps(second) == transformed_first.overlaps(transformed_second)
    origin, direction = Vec2(-10, 0), Vec2(1, 0)
    original_range = ray_rectangle_distance(origin, direction, first)
    transformed_range = ray_rectangle_distance(
        transform.local_to_world(origin), direction.rotate(transform.yaw), transformed_first
    )
    assert transformed_range == pytest.approx(original_range)


def test_quintic_derivative_matches_finite_difference():
    polynomial = Quintic(Boundary(-1, 0.2), Boundary(2, -0.1), 4)
    positions = np.linspace(0.2, 3.8, 30)
    epsilon = 1e-5
    numerical = (polynomial.value(positions + epsilon) - polynomial.value(positions - epsilon)) / (
        2 * epsilon
    )
    np.testing.assert_allclose(numerical, polynomial.value(positions, 1), rtol=1e-6, atol=1e-8)


def test_meter_batch_partition_does_not_change_results():
    generator = torch.Generator().manual_seed(99)
    target = torch.randint(0, 3, (7, 9), generator=generator)
    prediction = torch.randint(0, 3, (7, 9), generator=generator)
    complete, partitioned = ConfusionMeter(), ConfusionMeter()
    complete.update(prediction, target)
    for rows in (slice(0, 2), slice(2, 6), slice(6, 7)):
        partitioned.update(prediction[rows], target[rows])
    assert complete.summary() == partitioned.summary()
    assert complete.summary().evaluated_pixels == 63


def test_repeated_simulation_produces_identical_states():
    scenario = replace(make_scenario("lane_following"), duration=2)
    first = run_episode(scenario, Driver("pursuit"))
    second = run_episode(scenario, Driver("pursuit"))
    assert first == second
    times = [frame.state.time for frame in first.frames]
    assert all(b > a for a, b in zip(times[:-1], times[1:], strict=True))


def test_nonintegral_duration_has_no_overshoot():
    scenario = replace(make_scenario("lane_following"), duration=0.27, dt=0.1)
    episode = run_episode(scenario)
    assert episode.final.time == pytest.approx(0.27)
    assert max(frame.state.time for frame in episode.frames) <= 0.27 + 1e-12


def test_small_time_step_change_preserves_straight_motion():
    initial = VehicleState(speed=4)
    fine, coarse = initial, initial
    for _ in range(100):
        fine = bicycle_step(fine, Control(1, 0), 0.01)
    for _ in range(10):
        coarse = bicycle_step(coarse, Control(1, 0), 0.1)
    assert fine.pose.x == pytest.approx(coarse.pose.x, abs=1e-10)
    assert fine.speed == pytest.approx(coarse.speed, abs=1e-10)
