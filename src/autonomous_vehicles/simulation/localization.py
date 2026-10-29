"""Linear constant-velocity Kalman localization in a metric world frame.

The state is [x, y, vx, vy]. Measurements contain world x and y. This model
is intentionally independent of the vehicle's nonlinear bicycle dynamics.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LocalizationConfig:
    acceleration_noise: float = 0.5
    position_noise: float = 0.3
    initial_position_std: float = 1.0
    initial_velocity_std: float = 3.0
    gating_squared_distance: float = 9.21

    def __post_init__(self):
        values = np.array(list(vars(self).values()))
        if not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError("Localization noise and gating parameters must be finite and positive")


@dataclass(frozen=True)
class LocalizationEstimate:
    state: np.ndarray
    covariance: np.ndarray
    accepted: bool
    innovation_squared_distance: float | None

    @property
    def position(self) -> np.ndarray:
        return self.state[:2].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.state[2:].copy()


def symmetric(matrix: np.ndarray) -> np.ndarray:
    return (matrix + matrix.T) / 2


class ConstantVelocityFilter:
    def __init__(self, position=(0.0, 0.0), config: LocalizationConfig | None = None):
        self.config = LocalizationConfig() if config is None else config
        position = np.asarray(position, dtype=float)
        if position.shape != (2,) or not np.isfinite(position).all():
            raise ValueError("Initial position must be a finite XY vector")
        self.state = np.concatenate((position, [0, 0]))
        self.covariance = np.diag(
            [
                self.config.initial_position_std**2,
                self.config.initial_position_std**2,
                self.config.initial_velocity_std**2,
                self.config.initial_velocity_std**2,
            ]
        )
        self.observation_matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        self.measurement_covariance = np.eye(2) * self.config.position_noise**2

    def snapshot(
        self, accepted: bool = True, distance: float | None = None
    ) -> LocalizationEstimate:
        return LocalizationEstimate(self.state.copy(), self.covariance.copy(), accepted, distance)

    def predict(self, dt: float) -> LocalizationEstimate:
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Prediction interval must be finite and positive")
        transition = np.array(
            [
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=float,
        )
        influence = np.array([[dt**2 / 2, 0], [0, dt**2 / 2], [dt, 0], [0, dt]])
        process = influence @ influence.T * self.config.acceleration_noise**2
        self.state = transition @ self.state
        self.covariance = symmetric(transition @ self.covariance @ transition.T + process)
        return self.snapshot()

    def update(self, measurement) -> LocalizationEstimate:
        measurement = np.asarray(measurement, dtype=float)
        if measurement.shape != (2,) or not np.isfinite(measurement).all():
            raise ValueError("Measurement must be a finite XY vector")
        observation = self.observation_matrix
        innovation = measurement - observation @ self.state
        innovation_covariance = (
            observation @ self.covariance @ observation.T + self.measurement_covariance
        )
        distance = float(innovation @ np.linalg.solve(innovation_covariance, innovation))
        if distance > self.config.gating_squared_distance:
            return self.snapshot(False, distance)
        gain = np.linalg.solve(innovation_covariance, observation @ self.covariance).T
        self.state = self.state + gain @ innovation
        residual = np.eye(4) - gain @ observation
        # Joseph covariance update preserves positive semidefiniteness numerically.
        self.covariance = symmetric(
            residual @ self.covariance @ residual.T + gain @ self.measurement_covariance @ gain.T
        )
        return self.snapshot(True, distance)


def noisy_positions(positions, standard_deviation: float = 0.3, seed: int = 42) -> np.ndarray:
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 2 or not np.isfinite(positions).all():
        raise ValueError("Expected finite Nx2 positions")
    if not np.isfinite(standard_deviation) or standard_deviation < 0 or seed < 0:
        raise ValueError("Invalid measurement noise configuration")
    return positions + np.random.default_rng(seed).normal(0, standard_deviation, positions.shape)
