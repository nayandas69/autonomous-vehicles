"""Time-aware PID with conditional integral anti-windup and reset."""

from dataclasses import dataclass

from autonomous_vehicles.simulation.geometry import clamp, finite


@dataclass(frozen=True)
class PIDConfig:
    kp: float = 1.2
    ki: float = 0.15
    kd: float = 0.05
    lower: float = -7.0
    upper: float = 3.0
    integral_limit: float = 10.0
    derivative_smoothing: float = 0.2

    def __post_init__(self):
        finite(*vars(self).values())
        if min(self.kp, self.ki, self.kd) < 0 or self.lower >= self.upper:
            raise ValueError("Gains must be nonnegative and output bounds ordered")
        if self.integral_limit < 0 or not 0 <= self.derivative_smoothing <= 1:
            raise ValueError("Invalid integral limit or smoothing factor")


class PID:
    def __init__(self, config: PIDConfig | None = None):
        self.config = PIDConfig() if config is None else config
        self.reset()

    def reset(self) -> None:
        self.integral = 0.0
        self.previous_error = None
        self.derivative = 0.0

    def step(self, error: float, dt: float) -> float:
        finite(error, dt)
        if dt <= 0:
            raise ValueError("dt must be positive")
        raw_derivative = 0 if self.previous_error is None else (error - self.previous_error) / dt
        alpha = self.config.derivative_smoothing
        self.derivative = alpha * raw_derivative + (1 - alpha) * self.derivative
        proposed = clamp(
            self.integral + error * dt,
            -self.config.integral_limit,
            self.config.integral_limit,
        )
        raw = self.config.kp * error + self.config.ki * proposed + self.config.kd * self.derivative
        output = clamp(raw, self.config.lower, self.config.upper)
        # Accept integration if unsaturated or if integration would reduce saturation.
        if raw == output or (raw > output and error < 0) or (raw < output and error > 0):
            self.integral = proposed
        self.previous_error = error
        return output


class SpeedController:
    def __init__(self, config: PIDConfig | None = None):
        self.pid = PID(config)

    def reset(self) -> None:
        self.pid.reset()

    def command(self, target: float, measured: float, dt: float) -> float:
        finite(target, measured)
        if target < 0 or measured < 0:
            raise ValueError("Speeds must be nonnegative")
        return self.pid.step(target - measured, dt)
