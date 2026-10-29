"""Boundary-constrained quintic lateral transitions and derivatives."""

from dataclasses import dataclass

import numpy as np

from autonomous_vehicles.simulation.geometry import finite


@dataclass(frozen=True)
class Boundary:
    position: float
    velocity: float = 0.0
    acceleration: float = 0.0

    def __post_init__(self):
        finite(self.position, self.velocity, self.acceleration)


class Quintic:
    """Match position, first derivative and second derivative at both ends."""

    def __init__(self, start: Boundary, end: Boundary, duration: float):
        finite(duration)
        if duration <= 1e-6:
            raise ValueError("Duration must exceed one microsecond")
        self.duration = duration
        t = duration
        first = np.array([start.position, start.velocity, start.acceleration / 2])
        matrix = np.array(
            [
                [t**3, t**4, t**5],
                [3 * t**2, 4 * t**3, 5 * t**4],
                [6 * t, 12 * t**2, 20 * t**3],
            ]
        )
        rhs = np.array(
            [
                end.position - first[0] - first[1] * t - first[2] * t**2,
                end.velocity - first[1] - 2 * first[2] * t,
                end.acceleration - 2 * first[2],
            ]
        )
        self.coefficients = np.concatenate((first, np.linalg.solve(matrix, rhs)))
        self.coefficients.setflags(write=False)

    def value(self, time, derivative: int = 0):
        if derivative < 0 or derivative > 5:
            raise ValueError("Derivative order must be between zero and five")
        values = np.asarray(time, dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Evaluation coordinates must be finite")
        coefficients = self.coefficients.copy()
        for _ in range(derivative):
            coefficients = np.array([index * value for index, value in enumerate(coefficients)][1:])
        return np.polynomial.polynomial.polyval(values, coefficients)

    def sample(self, count: int = 51) -> tuple[np.ndarray, np.ndarray]:
        if count < 2:
            raise ValueError("Need at least two samples")
        times = np.linspace(0, self.duration, count)
        return times, self.value(times)

    def integrated_squared_jerk(self, count: int = 101) -> float:
        if count < 2:
            raise ValueError("Need at least two integration points")
        times = np.linspace(0, self.duration, count)
        return float(np.trapezoid(self.value(times, derivative=3) ** 2, times))
