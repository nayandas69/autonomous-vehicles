"""Plotting smoke checks with headless backend and explicit unit labels."""

from dataclasses import replace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
import torch

from autonomous_vehicles.data import RoadScenes
from autonomous_vehicles.perception.calibration import calibration_report
from autonomous_vehicles.planning.lattice import LatticePlanner
from autonomous_vehicles.simulation.scenarios import make_scenario
from autonomous_vehicles.simulation.sensors import RangeSensor
from autonomous_vehicles.simulation.world import run_episode
from autonomous_vehicles.visualization import (
    plot_calibration,
    plot_candidates,
    plot_episode,
    plot_history,
    plot_scan,
    plot_scene,
    plot_segmentation,
)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_scene_and_episode_have_units_and_four_panels():
    scenario = replace(make_scenario("lane_following"), duration=0.2)
    episode = run_episode(scenario)
    ax = plot_scene(scenario)
    assert ax.get_xlabel() == "World x (m)"
    assert ax.get_ylabel() == "World y (m)"
    figure = plot_episode(episode, scenario)
    assert len(figure.axes) == 4
    assert "Speed (m/s)" in [ax.get_ylabel() for ax in figure.axes]


def test_segmentation_uses_fixed_semantic_palette():
    image, labels = RoadScenes(size=32)[0]
    figure = plot_segmentation(image, labels, labels)
    assert len(figure.axes) == 3
    assert [ax.get_title() for ax in figure.axes] == ["RGB", "Ground truth", "Prediction"]
    assert figure.axes[1].images[0].get_clim() == (0, 2)
    assert figure.axes[2].images[0].get_clim() == (0, 2)


def test_history_and_candidate_views_render_data():
    history = [
        {"epoch": 1, "train_loss": 1, "mean_iou": 0.2},
        {"epoch": 2, "train_loss": 0.8, "mean_iou": 0.3},
    ]
    figure = plot_history(history)
    assert len(figure.axes[0].lines[0].get_ydata()) == 2
    road = make_scenario("lane_following").road
    candidates = LatticePlanner().candidates(road, 5)
    figure = plot_candidates(candidates, road)
    assert len(figure.axes[0].lines) == len(candidates) + 1
    with pytest.raises(ValueError):
        plot_history([])


def test_calibration_and_range_plot_smoke():
    report = calibration_report(torch.zeros(1, 3, 2, 2), torch.zeros(1, 2, 2, dtype=torch.int64))
    figure = plot_calibration(report)
    assert figure.axes[0].get_xlim() == (0, 1)
    scenario = make_scenario("stopped_vehicle")
    scan = RangeSensor().scan(scenario.initial.pose, scenario.actors_at(0))
    ax = plot_scan(scan)
    assert ax.get_xlabel() == "x (m)"
    assert len(ax.collections) == 2
