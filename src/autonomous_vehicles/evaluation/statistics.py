"""Seeded bootstrap uncertainty and paired research comparisons."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Estimate:
    mean: float
    lower: float
    upper: float
    observations: int
    confidence: float


def values_array(values) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0 or not np.isfinite(array).all():
        raise ValueError("Expected a nonempty finite vector")
    return array


def bootstrap_mean(
    values, confidence: float = 0.95, resamples: int = 2000, seed: int = 42
) -> Estimate:
    array = values_array(values)
    if not 0 < confidence < 1 or resamples < 2 or seed < 0:
        raise ValueError("Invalid bootstrap configuration")
    rng = np.random.default_rng(seed)
    means = np.empty(resamples)
    # Sample one replicate at a time to bound memory for large result vectors.
    for index in range(resamples):
        means[index] = rng.choice(array, size=len(array), replace=True).mean()
    tail = (1 - confidence) / 2
    lower, upper = np.quantile(means, [tail, 1 - tail])
    return Estimate(float(array.mean()), float(lower), float(upper), len(array), confidence)


def paired_difference(first, second, **kwargs) -> Estimate:
    first, second = values_array(first), values_array(second)
    if first.shape != second.shape:
        raise ValueError("Paired observations must have equal shape and matching identities")
    return bootstrap_mean(first - second, **kwargs)


def wilson_interval(
    successes: int, trials: int, z: float = 1.959963984540054
) -> tuple[float, float]:
    if trials < 1 or not 0 <= successes <= trials or not 0 < z < float("inf"):
        raise ValueError("Invalid binomial counts or normal quantile")
    proportion = successes / trials
    denominator = 1 + z * z / trials
    center = (proportion + z * z / (2 * trials)) / denominator
    half = (
        z * np.sqrt(proportion * (1 - proportion) / trials + z * z / (4 * trials**2)) / denominator
    )
    return max(0.0, float(center - half)), min(1.0, float(center + half))


def quantile_summary(values) -> dict:
    array = values_array(values)
    return {
        "count": len(array),
        "minimum": float(array.min()),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "p95": float(np.quantile(array, 0.95)),
        "maximum": float(array.max()),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
    }


def moving_average(values, window: int = 5) -> np.ndarray:
    array = values_array(values)
    if window < 1 or window > len(array):
        raise ValueError("Window must lie between one and observation count")
    return np.convolve(array, np.ones(window) / window, mode="valid")
