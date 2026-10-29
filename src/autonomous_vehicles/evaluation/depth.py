"""Depth metrics with units, validity masking, and scale alignment kept explicit."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DepthMetrics:
    valid_pixels: int
    mean_absolute_error: float
    root_mean_squared_error: float
    absolute_relative_error: float
    log_rmse: float
    delta_1: float
    delta_2: float
    delta_3: float


def valid_depth_arrays(
    prediction, target, valid=None, minimum: float = 0.1, maximum: float = 100.0
):
    prediction, target = np.asarray(prediction, dtype=float), np.asarray(target, dtype=float)
    if prediction.shape != target.shape or prediction.ndim < 1:
        raise ValueError("Prediction and target shapes must match")
    if not 0 < minimum < maximum < float("inf"):
        raise ValueError("Depth limits must be finite and ordered above zero")
    selected = np.isfinite(target) & (target >= minimum) & (target <= maximum)
    if valid is not None:
        valid = np.asarray(valid)
        if valid.shape != target.shape or valid.dtype != bool:
            raise ValueError("Validity mask must be boolean and align with target")
        selected &= valid
    if not selected.any():
        raise ValueError("No valid target depths")
    if not np.isfinite(prediction[selected]).all() or np.any(prediction[selected] <= 0):
        raise ValueError("Predicted depths at valid targets must be finite and positive")
    return prediction[selected], target[selected]


def depth_metrics(
    prediction, target, valid=None, minimum: float = 0.1, maximum: float = 100.0
) -> DepthMetrics:
    prediction, target = valid_depth_arrays(prediction, target, valid, minimum, maximum)
    error = prediction - target
    ratio = np.maximum(prediction / target, target / prediction)
    log_error = np.log(prediction) - np.log(target)
    return DepthMetrics(
        len(target),
        float(np.mean(np.abs(error))),
        float(np.sqrt(np.mean(error**2))),
        float(np.mean(np.abs(error) / target)),
        float(np.sqrt(np.mean(log_error**2))),
        float(np.mean(ratio < 1.25)),
        float(np.mean(ratio < 1.25**2)),
        float(np.mean(ratio < 1.25**3)),
    )


def median_scale(prediction, target, valid=None) -> float:
    """Use only for explicitly scale-ambiguous experiments; report the fitted scale."""
    prediction, target = valid_depth_arrays(prediction, target, valid)
    return float(np.median(target) / np.median(prediction))
