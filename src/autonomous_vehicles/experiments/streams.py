"""Timestamped observation buffering for offline sensor synchronization studies.

The buffer is bounded, rejects out-of-order insertions and only exposes samples
at or before a query time. No future observation can leak into the current frame.
"""

from bisect import bisect_right
from dataclasses import dataclass
from math import isfinite
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class TimedObservation(Generic[T]):
    time: float
    value: T

    def __post_init__(self):
        if not isfinite(self.time) or self.time < 0:
            raise ValueError("Observation timestamp must be finite and nonnegative")


class ObservationBuffer(Generic[T]):
    def __init__(self, capacity: int = 100):
        if capacity < 1:
            raise ValueError("Buffer capacity must be positive")
        self.capacity = capacity
        self.observations: list[TimedObservation[T]] = []

    def __len__(self) -> int:
        return len(self.observations)

    def clear(self) -> None:
        self.observations.clear()

    def append(self, observation: TimedObservation[T]) -> None:
        if self.observations and observation.time <= self.observations[-1].time:
            raise ValueError("Observations must have strictly increasing timestamps")
        self.observations.append(observation)
        if len(self.observations) > self.capacity:
            del self.observations[0]

    def latest_before(
        self, time: float, maximum_age: float | None = None
    ) -> TimedObservation[T] | None:
        if not isfinite(time) or time < 0:
            raise ValueError("Query time must be finite and nonnegative")
        if maximum_age is not None and (not isfinite(maximum_age) or maximum_age < 0):
            raise ValueError("Maximum age must be finite and nonnegative")
        index = bisect_right([observation.time for observation in self.observations], time) - 1
        if index < 0:
            return None
        observation = self.observations[index]
        if maximum_age is not None and time - observation.time > maximum_age:
            return None
        return observation

    def between(self, start: float, end: float) -> tuple[TimedObservation[T], ...]:
        if not isfinite(start) or not isfinite(end) or not 0 <= start <= end:
            raise ValueError("Interval must be finite and ordered above zero")
        return tuple(
            observation for observation in self.observations if start <= observation.time <= end
        )


def synchronize(buffers: dict[str, ObservationBuffer], time: float, maximum_age: float) -> dict:
    if not buffers:
        raise ValueError("Need at least one sensor buffer")
    observations = {
        name: buffer.latest_before(time, maximum_age) for name, buffer in buffers.items()
    }
    return {
        "time": time,
        "ready": all(observation is not None for observation in observations.values()),
        "values": {
            name: observation.value if observation is not None else None
            for name, observation in observations.items()
        },
        "ages": {
            name: time - observation.time if observation is not None else None
            for name, observation in observations.items()
        },
    }
