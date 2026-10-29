"""Controller signs, actuator saturation and time-based regulator behavior."""

import pytest

from autonomous_vehicles.control.driver import Driver
from autonomous_vehicles.control.lateral import PurePursuit, PursuitConfig, Stanley, StanleyConfig
from autonomous_vehicles.control.pid import PID, PIDConfig, SpeedController
from autonomous_vehicles.control.safety import SafetyConfig, StoppingEnvelope
from autonomous_vehicles.simulation.actors import static_obstacle
from autonomous_vehicles.simulation.dynamics import VehicleParameters, VehicleState
from autonomous_vehicles.simulation.geometry import Pose2
from autonomous_vehicles.simulation.roads import straight_road


def test_pid_proportional_reference():
    pid = PID(PIDConfig(kp=2, ki=0, kd=0, lower=-100, upper=100))
    assert pid.step(3, 0.1) == 6
    assert pid.step(-2, 0.1) == -4


def test_pid_integral_uses_seconds():
    pid = PID(PIDConfig(kp=0, ki=1, kd=0, lower=-100, upper=100))
    assert pid.step(2, 0.5) == 1
    assert pid.step(2, 0.5) == 2
    pid.reset()
    assert pid.integral == 0
    assert pid.previous_error is None


def test_pid_anti_windup_does_not_accumulate_saturation():
    pid = PID(PIDConfig(kp=1, ki=1, kd=0, lower=-1, upper=1))
    for _ in range(100):
        assert pid.step(100, 0.1) == 1
    assert pid.integral == 0
    assert pid.step(0, 0.1) == 0


def test_derivative_has_no_first_step_kick():
    pid = PID(PIDConfig(kp=0, ki=0, kd=1, lower=-100, upper=100, derivative_smoothing=1))
    assert pid.step(5, 0.1) == 0
    assert pid.step(6, 0.1) == pytest.approx(10)


@pytest.mark.parametrize("controller", [PurePursuit(), Stanley()])
def test_lateral_control_returns_toward_reference(controller):
    parameters = VehicleParameters()
    road = straight_road()
    above = VehicleState(Pose2(10, 1), speed=5)
    below = VehicleState(Pose2(10, -1), speed=5)
    assert controller.command(above, road, parameters) < 0
    assert controller.command(below, road, parameters) > 0
    centered = VehicleState(Pose2(10, 0), speed=5)
    assert controller.command(centered, road, parameters) == pytest.approx(0)


def test_safety_clear_and_blocked_corridor():
    envelope = StoppingEnvelope()
    parameters = VehicleParameters()
    state = VehicleState(Pose2(), speed=8)
    clear = envelope.decide(state, (), 10, parameters)
    assert clear.target_speed == 10
    assert not clear.emergency
    near = static_obstacle("near", 8, 0).at(0)
    blocked = envelope.decide(state, (near,), 10, parameters)
    assert blocked.emergency
    assert blocked.target_speed < clear.target_speed
    assert blocked.actor_id == "near"


def test_safety_ignores_distant_lateral_actor_and_rear_actor():
    envelope = StoppingEnvelope()
    parameters = VehicleParameters()
    actors = (static_obstacle("side", 20, 20).at(0), static_obstacle("rear", -10, 0).at(0))
    result = envelope.decide(VehicleState(speed=5), actors, 8, parameters)
    assert result.reason == "clear"


def test_safety_allowed_speed_grows_with_gap():
    envelope = StoppingEnvelope()
    parameters = VehicleParameters()
    state = VehicleState(speed=3)
    speeds = [
        envelope.decide(
            state, (static_obstacle("front", gap, 0).at(0),), 20, parameters
        ).target_speed
        for gap in (10, 20, 40)
    ]
    assert speeds == sorted(speeds)
    assert len(set(speeds)) == 3


def test_driver_emergency_commands_full_braking():
    parameters = VehicleParameters()
    state = VehicleState(Pose2(2, 0), speed=10)
    output = Driver().command(
        state, straight_road(), (static_obstacle("near", 9, 0).at(0),), 10, 0.1, parameters
    )
    assert output.decision.emergency
    assert output.control.acceleration == -parameters.max_deceleration


def test_driver_reset_clears_speed_state():
    driver = Driver()
    driver.speed.pid.integral = 5
    driver.reset()
    assert driver.speed.pid.integral == 0
    with pytest.raises(ValueError):
        Driver("missing")


@pytest.mark.parametrize(
    "factory",
    [
        lambda: PIDConfig(kp=-1),
        lambda: PIDConfig(lower=1, upper=0),
        lambda: PursuitConfig(minimum_lookahead=0),
        lambda: StanleyConfig(softening_speed=0),
        lambda: SafetyConfig(comfortable_deceleration=0),
        lambda: SpeedController().command(-1, 1, 0.1),
        lambda: PID().step(1, 0),
    ],
)
def test_invalid_controller_settings(factory):
    with pytest.raises(ValueError):
        factory()
