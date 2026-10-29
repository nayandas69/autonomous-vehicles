"""Physical bounds, stationary behavior, and analytical straight motion."""

from math import atan

import pytest

from autonomous_vehicles.simulation.dynamics import (
    Control,
    VehicleParameters,
    VehicleState,
    bicycle_step,
    steering_for_curvature,
    stopping_distance,
)
from autonomous_vehicles.simulation.geometry import Pose2


def test_straight_constant_speed_motion():
    state = VehicleState(Pose2(1, 2), speed=5)
    result = bicycle_step(state, Control(), 0.1)
    assert result.pose.x == pytest.approx(1.5)
    assert result.pose.y == pytest.approx(2)
    assert result.pose.yaw == 0
    assert result.speed == 5
    assert result.time == pytest.approx(0.1)


def test_constant_acceleration_matches_distance_equation():
    state = VehicleState(speed=2)
    result = bicycle_step(state, Control(2, 0), 0.2)
    assert result.speed == pytest.approx(2.4)
    assert result.pose.x == pytest.approx(2 * 0.2 + 0.5 * 2 * 0.2**2)
    assert result.acceleration == pytest.approx(2)


def test_braking_never_reverses_vehicle():
    state = VehicleState(speed=0.1)
    result = bicycle_step(state, Control(-100, 0), 0.1)
    assert result.speed == 0
    next_state = bicycle_step(result, Control(-7, 0), 0.1)
    assert next_state.pose == result.pose
    assert next_state.speed == 0


def test_speed_acceleration_and_steering_are_bounded():
    parameters = VehicleParameters(max_speed=4)
    state = VehicleState(speed=3.9)
    for _ in range(20):
        next_state = bicycle_step(state, Control(100, 10), 0.1, parameters)
        assert next_state.speed <= parameters.max_speed
        assert abs(next_state.steering) <= parameters.max_steering
        assert (
            abs(next_state.steering - state.steering) <= parameters.max_steering_rate * 0.1 + 1e-12
        )
        assert next_state.acceleration <= parameters.max_acceleration + 1e-12
        state = next_state


def test_positive_steering_turns_left():
    state = VehicleState(speed=5)
    for _ in range(10):
        state = bicycle_step(state, Control(0, 0.3), 0.1)
    assert state.pose.y > 0
    assert state.pose.yaw > 0


def test_vehicle_footprint_uses_rear_axle_reference():
    parameters = VehicleParameters()
    body = VehicleState(Pose2(10, 0)).footprint(parameters)
    assert body.aabb()[0] == pytest.approx(10 - parameters.rear_overhang)
    assert body.aabb()[2] == pytest.approx(10 + parameters.length - parameters.rear_overhang)


def test_curvature_to_steering_uses_wheelbase():
    parameters = VehicleParameters()
    assert steering_for_curvature(0.1, parameters) == pytest.approx(atan(0.27))
    assert steering_for_curvature(100, parameters) == parameters.max_steering


def test_stopping_distance_includes_reaction_time():
    assert stopping_distance(10, 5) == 10
    assert stopping_distance(10, 5, 0.5) == 15
    assert stopping_distance(0, 5) == 0


@pytest.mark.parametrize("dt", [0, -0.1, 0.6, float("nan")])
def test_invalid_time_steps(dt):
    with pytest.raises(ValueError):
        bicycle_step(VehicleState(), Control(), dt)


@pytest.mark.parametrize("kwargs", [{"width": 0}, {"length": 1}, {"max_steering": 2}])
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        VehicleParameters(**kwargs)


def test_invalid_state_and_stopping_arguments():
    with pytest.raises(ValueError):
        VehicleState(speed=-1)
    with pytest.raises(ValueError):
        stopping_distance(2, 0)
    with pytest.raises(ValueError):
        Control(float("inf"), 0)
