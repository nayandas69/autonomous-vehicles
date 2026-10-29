"""Boundary-constrained paths, obstacle rejection, and speed feasibility."""

from math import pi

import numpy as np
import pytest

from autonomous_vehicles.planning.lattice import LatticeConfig, LatticePlanner
from autonomous_vehicles.planning.path import (
    Path,
    PathPoint,
    Trajectory,
    TrajectoryPoint,
    path_from_xy,
)
from autonomous_vehicles.planning.polynomial import Boundary, Quintic
from autonomous_vehicles.planning.prediction import (
    check_trajectory,
    predict_actor,
    time_to_closest_approach,
)
from autonomous_vehicles.planning.speed import (
    SpeedProfileConfig,
    plan_speed,
    trajectory_from_profile,
)
from autonomous_vehicles.simulation.actors import ActorState
from autonomous_vehicles.simulation.geometry import Pose2, Vec2
from autonomous_vehicles.simulation.roads import straight_road
from autonomous_vehicles.simulation.shapes import Rectangle


def test_quintic_satisfies_six_independent_boundary_constraints():
    start, end, duration = Boundary(1, 0.2, -0.1), Boundary(4, -0.3, 0.2), 5
    polynomial = Quintic(start, end, duration)
    for derivative, expected_start, expected_end in (
        (0, start.position, end.position),
        (1, start.velocity, end.velocity),
        (2, start.acceleration, end.acceleration),
    ):
        assert polynomial.value(0, derivative) == pytest.approx(expected_start)
        assert polynomial.value(duration, derivative) == pytest.approx(expected_end)


def test_quintic_constant_boundary_has_no_jerk():
    polynomial = Quintic(Boundary(2), Boundary(2), 3)
    times, values = polynomial.sample()
    np.testing.assert_allclose(values, 2)
    assert times[0] == 0 and times[-1] == 3
    assert polynomial.integrated_squared_jerk() == pytest.approx(0)


def test_slow_lateral_transition_reduces_jerk():
    fast = Quintic(Boundary(0), Boundary(2), 2)
    slow = Quintic(Boundary(0), Boundary(2), 4)
    assert slow.integrated_squared_jerk() < fast.integrated_squared_jerk()


def test_path_interpolation_and_nearest_reference():
    path = path_from_xy(np.array([[0, 0], [3, 0], [6, 0]]))
    assert path.length == 6
    assert path.pose_at(1.5).x == pytest.approx(1.5)
    assert path.pose_at(-1).x == 0
    assert path.pose_at(10).x == 6
    assert path.nearest(Vec2(2.9, 2)).s == 3
    assert all(point.curvature == 0 for point in path.points)


def test_trajectory_time_interpolation():
    trajectory = Trajectory(
        (
            TrajectoryPoint(0, Pose2(), 2),
            TrajectoryPoint(2, Pose2(6, 0), 4),
        )
    )
    midpoint = trajectory.sample(1)
    assert midpoint.pose.x == 3
    assert midpoint.speed == 3
    assert trajectory.duration == 2
    assert trajectory.sample(-1).time == 0
    assert trajectory.sample(10).time == 2


def test_lattice_prefers_center_when_unobstructed():
    planner = LatticePlanner()
    candidates = planner.candidates(straight_road(), 5)
    selected = planner.select(candidates)
    assert selected is not None
    assert selected.offset == 0
    assert selected.feasible
    assert selected.path.points[0].pose.position == Vec2(5, 0)


def test_lattice_reports_blocked_candidates_instead_of_unsafe_fallback():
    planner = LatticePlanner()
    obstacle = Rectangle(Pose2(20, 0), 4, 10)
    candidates = planner.candidates(straight_road(), 5, obstacles=(obstacle,))
    assert all(not candidate.feasible for candidate in candidates)
    assert planner.select(candidates) is None
    assert any(candidate.reason == "obstacle_clearance" for candidate in candidates)


def test_speed_profile_respects_energy_form_acceleration_bounds():
    path = path_from_xy(np.column_stack((np.linspace(0, 40, 81), np.zeros(81))))
    config = SpeedProfileConfig(maximum_speed=8, acceleration=2, deceleration=3)
    profile = plan_speed(path, config=config)
    assert profile.speeds[0] == profile.speeds[-1] == 0
    assert profile.speeds.max() <= 8
    ds = np.diff(profile.distances)
    acceleration = np.diff(profile.speeds**2) / (2 * ds)
    assert acceleration.max() <= 2 + 1e-10
    assert acceleration.min() >= -3 - 1e-10
    assert np.all(np.diff(profile.times) > 0)
    trajectory = trajectory_from_profile(path, profile)
    assert trajectory.duration == pytest.approx(profile.duration)


def test_curvature_limits_lateral_acceleration():
    angles = np.linspace(0, pi / 2, 101)
    path = path_from_xy(np.column_stack((10 * np.sin(angles), 10 * (1 - np.cos(angles)))))
    profile = plan_speed(path, config=SpeedProfileConfig(lateral_acceleration=1))
    lateral_acceleration = (
        np.array([abs(point.curvature) for point in path.points]) * profile.speeds**2
    )
    assert lateral_acceleration.max() <= 1 + 1e-9


def test_prediction_and_sampled_collision_detection():
    actor = ActorState("moving", Pose2(10, 0), 2)
    assert predict_actor(actor, 2).pose.x == 14
    trajectory = Trajectory((TrajectoryPoint(0, Pose2(), 5), TrajectoryPoint(2, Pose2(10, 0), 5)))
    stopped = ActorState("stopped", Pose2(10, 0), 0)
    risk = check_trajectory(trajectory, (stopped,))
    assert risk.collision and risk.actor_id == "stopped"
    assert risk.earliest_time == 2
    clear = check_trajectory(trajectory, (ActorState("side", Pose2(10, 10), 0),))
    assert not clear.collision


def test_closest_approach_is_relative_motion():
    first = ActorState("a", Pose2(0, 0), 2)
    second = ActorState("b", Pose2(10, 0, pi), 2)
    time, distance = time_to_closest_approach(first, second)
    assert time == pytest.approx(2.5)
    assert distance == pytest.approx(0, abs=1e-12)
    time, distance = time_to_closest_approach(first, second, horizon=1)
    assert time == 1
    assert distance == pytest.approx(6)


@pytest.mark.parametrize(
    "operation",
    [
        lambda: Quintic(Boundary(0), Boundary(1), 0),
        lambda: Quintic(Boundary(0), Boundary(1), 1).value(1, 6),
        lambda: path_from_xy(np.array([[0, 0], [0, 0]])),
        lambda: Path((PathPoint(Pose2(), 1), PathPoint(Pose2(1, 0), 2))),
        lambda: LatticeConfig(offsets=()),
        lambda: LatticeConfig(offsets=(0, 0)),
        lambda: SpeedProfileConfig(acceleration=0),
        lambda: LatticePlanner().candidates(straight_road(), 200),
    ],
)
def test_planning_rejects_invalid_contracts(operation):
    with pytest.raises(ValueError):
        operation()
