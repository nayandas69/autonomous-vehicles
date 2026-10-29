"""Join experiment records by scenario and controller before comparing runs."""

from dataclasses import dataclass

from autonomous_vehicles.evaluation.statistics import Estimate, paired_difference


@dataclass(frozen=True)
class ExperimentComparison:
    metric: str
    keys: tuple[tuple[str, str, int], ...]
    difference: Estimate


def indexed_records(report: dict) -> dict:
    if report.get("format_version") != 1 or not isinstance(report.get("records"), list):
        raise ValueError("Unsupported simulation report")
    records = {}
    for record in report["records"]:
        key = (record["metrics"]["scenario"], record["controller"], record["repetition"])
        if key in records:
            raise ValueError("Duplicate experiment record")
        records[key] = record["metrics"]
    return records


def compare_reports(
    first: dict, second: dict, metric: str, resamples: int = 2000
) -> ExperimentComparison:
    left, right = indexed_records(first), indexed_records(second)
    if set(left) != set(right) or not left:
        raise ValueError("Comparison requires exactly matching experiment identities")
    keys = tuple(sorted(left))
    try:
        left_values = [float(left[key][metric]) for key in keys]
        right_values = [float(right[key][metric]) for key in keys]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Metric {metric!r} is missing or not numeric") from exc
    estimate = paired_difference(left_values, right_values, resamples=resamples)
    return ExperimentComparison(metric, keys, estimate)


def leaderboard(
    report: dict, metric: str = "lateral_rmse_meters", lower_is_better: bool = True
) -> list[dict]:
    records = indexed_records(report)
    groups = {}
    for (_, controller, _), result in records.items():
        value = result.get(metric)
        if value is None or not isinstance(value, (int, float)):
            raise ValueError(f"Metric {metric!r} unavailable")
        groups.setdefault(controller, []).append(value)
    rows = [
        {
            "controller": controller,
            "metric": metric,
            "mean": sum(values) / len(values),
            "episodes": len(values),
        }
        for controller, values in groups.items()
    ]
    return sorted(rows, key=lambda row: row["mean"], reverse=not lower_is_better)
