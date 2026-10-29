"""Experiment serialization, configuration, uncertainty and RNG isolation."""

import json
import random

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from autonomous_vehicles.data import RoadScenes
from autonomous_vehicles.evaluation.statistics import (
    bootstrap_mean,
    moving_average,
    paired_difference,
    quantile_summary,
    wilson_interval,
)
from autonomous_vehicles.experiments.artifacts import (
    atomic_json,
    file_sha256,
    source_fingerprint,
    stable_identifier,
)
from autonomous_vehicles.experiments.comparison import compare_reports, leaderboard
from autonomous_vehicles.experiments.config import SimulationConfig
from autonomous_vehicles.experiments.reproducibility import reproducible, seed_for
from autonomous_vehicles.experiments.sweep import SweepTrial, rank_settings
from autonomous_vehicles.experiments.training import EarlyStopping, choose_loss, train_epoch
from autonomous_vehicles.model import RoadNet


def test_atomic_json_refuses_overwrite_and_invalid_json_numbers(tmp_path):
    path = tmp_path / "report.json"
    atomic_json(path, {"value": np.array([1, 2])})
    assert json.loads(path.read_text()) == {"value": [1, 2]}
    with pytest.raises(FileExistsError):
        atomic_json(path, {"value": 3})
    assert json.loads(path.read_text()) == {"value": [1, 2]}
    atomic_json(path, {"value": 4}, overwrite=True)
    assert json.loads(path.read_text()) == {"value": 4}
    with pytest.raises(ValueError):
        atomic_json(tmp_path / "invalid.json", {"value": float("nan")})
    assert not (tmp_path / "invalid.json").exists()


def test_source_fingerprints_track_content_not_timestamps(tmp_path):
    source = tmp_path / "sample.py"
    source.write_text("x = 1\n")
    first = source_fingerprint(tmp_path)
    assert first == {"sample.py": file_sha256(source)}
    source.write_text("x = 2\n")
    assert source_fingerprint(tmp_path) != first
    assert stable_identifier({"a": 1, "b": 2}) == stable_identifier({"b": 2, "a": 1})


def test_configuration_roundtrip_and_unknown_fields(tmp_path):
    config = SimulationConfig()
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config.to_dict()))
    assert SimulationConfig.load(path) == config
    with pytest.raises(ValueError, match="Unknown"):
        SimulationConfig.from_dict({"mistyped": 1})
    with pytest.raises(ValueError):
        SimulationConfig.from_dict({"safety": "false"})
    with pytest.raises(ValueError):
        SimulationConfig.from_dict({"scenarios": "lane_following"})


def test_bootstrap_repeatable_singleton_and_constant_values():
    first = bootstrap_mean([1, 2, 3], resamples=100, seed=7)
    second = bootstrap_mean([1, 2, 3], resamples=100, seed=7)
    assert first == second
    assert first.lower <= first.mean <= first.upper
    constant = bootstrap_mean([4, 4, 4], resamples=20)
    assert constant.mean == constant.lower == constant.upper == 4
    difference = paired_difference([2, 3], [1, 2], resamples=20)
    assert difference.mean == difference.lower == difference.upper == 1


def test_wilson_intervals_have_nonzero_uncertainty_at_extremes():
    lower, upper = wilson_interval(0, 10)
    assert lower == pytest.approx(0)
    assert 0 < upper < 1
    lower, upper = wilson_interval(10, 10)
    assert 0 < lower < 1
    assert upper == pytest.approx(1)


def test_statistical_summaries_and_invalid_inputs():
    summary = quantile_summary([1, 2, 3])
    assert summary["median"] == summary["mean"] == 2
    np.testing.assert_allclose(moving_average([1, 2, 3], 2), [1.5, 2.5])
    with pytest.raises(ValueError):
        bootstrap_mean([])
    with pytest.raises(ValueError):
        paired_difference([1], [1, 2])
    with pytest.raises(ValueError):
        wilson_interval(2, 1)


def test_scoped_reproducibility_restores_all_rng_states():
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.random.get_rng_state()
    with reproducible(7):
        first = (random.random(), np.random.random(), torch.rand(()))
    with reproducible(7):
        second = (random.random(), np.random.random(), torch.rand(()))
    assert first[0] == second[0]
    assert first[1] == second[1]
    torch.testing.assert_close(first[2], second[2])
    assert random.getstate() == python_before
    np.testing.assert_array_equal(np.random.get_state()[1], numpy_before[1])
    torch.testing.assert_close(torch.random.get_rng_state(), torch_before)
    assert seed_for(42, "train") == seed_for(42, "train")
    assert seed_for(42, "train") != seed_for(42, "test")


def test_early_stopping_patience_and_reset_on_improvement():
    stop = EarlyStopping(patience=2)
    assert not stop.update(0.2)
    assert not stop.update(0.2)
    assert not stop.update(0.3)
    assert not stop.update(0.3)
    assert stop.update(0.3)
    with pytest.raises(ValueError):
        stop.update(float("nan"))


def test_training_epoch_updates_parameters_and_returns_weighted_loss():
    with reproducible(42):
        model = RoadNet()
        before = [parameter.detach().clone() for parameter in model.parameters()]
        loader = DataLoader(RoadScenes(5, size=32), batch_size=2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        result = train_epoch(model, loader, optimizer)
        assert result.samples == 5 and result.batches == 3
        assert result.loss > 0 and result.gradient_norm_mean > 0
        assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters(), strict=True))
    with pytest.raises(ValueError):
        choose_loss("unknown")


def test_sweep_ranking_uses_validation_means():
    trials = (
        SweepTrial(1, 0.001, "dice", 0.5, 1),
        SweepTrial(2, 0.001, "dice", 0.7, 1),
        SweepTrial(1, 0.01, "dice", 0.4, 1),
    )
    ranked = rank_settings(trials)
    assert ranked[0]["learning_rate"] == 0.001
    assert ranked[0]["validation_iou_mean"] == pytest.approx(0.6)


def make_report(value):
    return {
        "format_version": 1,
        "records": [
            {
                "controller": "pursuit",
                "repetition": 0,
                "metrics": {"scenario": "lane_following", "error": value},
            }
        ],
    }


def test_report_comparison_joins_identities_and_rejects_unpaired_runs():
    first, second = make_report(2), make_report(1)
    result = compare_reports(first, second, "error", resamples=20)
    assert result.difference.mean == 1
    assert leaderboard(first, "error")[0]["mean"] == 2
    second["records"][0]["metrics"]["scenario"] = "different"
    with pytest.raises(ValueError, match="identities"):
        compare_reports(first, second, "error")
