"""Episode-level tracking, comfort and event metrics with explicit units."""

from dataclasses import asdict, dataclass

import numpy as np

from autonomous_vehicles.simulation.scenarios import Scenario
from autonomous_vehicles.simulation.world import Episode


@dataclass(frozen=True)
class DrivingMetrics:
    scenario: str
    termination: str
    duration_seconds: float
    distance_meters: float
    progress_meters: float
    collision: bool
    off_road: bool
    lateral_rmse_meters: float
    maximum_lateral_error_meters: float
    mean_speed_mps: float
    maximum_acceleration_mps2: float
    rms_jerk_mps3: float
    emergency_frames: int
    minimum_actor_clearance_meters: float | None

    def to_dict(self):
        return asdict(self)


def driving_metrics(episode: Episode, scenario: Scenario) -> DrivingMetrics:
    if not episode.frames or episode.scenario_name != scenario.name:
        raise ValueError("Episode must be nonempty and belong to scenario")
    frames = episode.frames
    times = np.array([frame.state.time for frame in frames])
    speeds = np.array([frame.state.speed for frame in frames])
    acceleration = np.array([frame.state.acceleration for frame in frames])
    lateral = np.array([frame.lateral_error for frame in frames])
    positions = np.array([[frame.state.pose.x, frame.state.pose.y] for frame in frames])
    distance = np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()
    differences = np.diff(times)
    if np.any(differences <= 0):
        raise ValueError("Episode timestamps must strictly increase")
    jerk = np.diff(acceleration) / differences if len(times) > 1 else np.array([0.0])
    clearances = []
    for frame in frames:
        ego = frame.state.footprint(scenario.parameters)
        for actor in frame.actors:
            # Circumscribed-disc lower bound; negative values do not prove collision.
            centers = (ego.pose.position - actor.pose.position).norm
            radius = (ego.length**2 + ego.width**2) ** 0.5 / 2
            actor_radius = (actor.length**2 + actor.width**2) ** 0.5 / 2
            clearances.append(centers - radius - actor_radius)
    progress = scenario.road.project(frames[-1].state.pose.position).s
    start = scenario.road.project(frames[0].state.pose.position).s
    duration = float(times[-1] - times[0])
    mean_speed = float(np.trapezoid(speeds, times) / duration) if duration > 0 else float(speeds[0])
    return DrivingMetrics(
        scenario.name,
        episode.termination,
        duration,
        float(distance),
        max(0, progress - start),
        episode.collided,
        episode.left_road,
        float(np.sqrt(np.mean(lateral**2))),
        float(np.max(np.abs(lateral))),
        mean_speed,
        float(np.max(np.abs(acceleration))),
        float(np.sqrt(np.mean(jerk**2))),
        sum(frame.emergency for frame in frames),
        min(clearances) if clearances else None,
    )


def aggregate_driving(metrics: tuple[DrivingMetrics, ...]) -> dict:
    if not metrics:
        raise ValueError("Need at least one episode")
    return {
        "episodes": len(metrics),
        "collision_rate": sum(metric.collision for metric in metrics) / len(metrics),
        "off_road_rate": sum(metric.off_road for metric in metrics) / len(metrics),
        "mean_lateral_rmse_meters": float(
            np.mean([metric.lateral_rmse_meters for metric in metrics])
        ),
        "mean_progress_meters": float(np.mean([metric.progress_meters for metric in metrics])),
        "mean_rms_jerk_mps3": float(np.mean([metric.rms_jerk_mps3 for metric in metrics])),
    }
