"""Installed command boundaries, run artifacts and actionable error exits."""

import json
import subprocess
import sys

import pytest


def command(*arguments):
    return subprocess.run(
        [sys.executable, "-m", "autonomous_vehicles", *arguments],
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_environment_and_scenario_listing():
    environment = command("environment")
    assert environment.returncode == 0
    payload = json.loads(environment.stdout)
    assert payload["python"].startswith("3.14")
    scenarios = command("scenarios")
    assert scenarios.returncode == 0
    assert "lane_following" in json.loads(scenarios.stdout)["scenarios"]


def test_cli_simulation_saves_report_and_frames(tmp_path):
    output = tmp_path / "run"
    result = command(
        "simulate", "--scenario", "lane_following", "--output", str(output), "--save-frames"
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["metrics"]["collision"] is False
    assert json.loads((output / "report.json").read_text()) == payload
    frames = json.loads((output / "frames.json").read_text())
    assert frames[0]["time"] == 0
    assert frames[-1]["time"] > 0
    duplicate = command("simulate", "--output", str(output))
    assert duplicate.returncode == 2
    assert "error:" in duplicate.stderr


def test_cli_benchmark_accepts_strict_config(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps({"scenarios": ["lane_following"], "controllers": ["stanley"], "repetitions": 1})
    )
    output = tmp_path / "suite"
    result = command("benchmark", "--config", str(config), "--output", str(output))
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["records"][0]["controller"] == "stanley"
    assert (output / "report.json").is_file()


@pytest.mark.parametrize(
    "arguments",
    [
        (),
        ("unknown",),
        ("train", "--output", "/tmp/unused", "--epochs", "0"),
        ("simulate", "--scenario", "missing", "--output", "/tmp/unused"),
        ("evaluate", "--checkpoint", "/tmp/no-such-av-checkpoint.pt"),
    ],
)
def test_cli_bad_arguments_exit_two(arguments):
    result = command(*arguments)
    assert result.returncode == 2
    assert "error" in result.stderr


def test_cli_missing_manifest_is_an_error(tmp_path):
    result = command("inspect-dataset", "--manifest", str(tmp_path / "missing.json"))
    assert result.returncode == 2
    assert "error:" in result.stderr
