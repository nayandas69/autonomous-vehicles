# Autonomous Vehicles

Original Python and PyTorch research software for **synthetic perception and driving
simulation**. The project includes a segmentation pipeline, a planar driving simulator,
controllers, sampled planning, sensor models, tracking, localization, evaluation and
executable Jupyter labs. All project implementations are written here.

This is a simulation research system. Its controllers use perfect simulator actor
observations; perception predictions are currently evaluated separately. Synthetic
results do not establish real-world driving performance.

[![checks](https://github.com/nayandas69/autonomous-vehicles/actions/workflows/ci.yml/badge.svg)](https://github.com/nayandas69/autonomous-vehicles/actions/workflows/ci.yml)

## Start

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
uv sync --locked --all-extras
uv run av scenarios
uv run av simulate --scenario stopped_vehicle --output artifacts/stopped --save-frames
uv run av benchmark --output artifacts/benchmark
uv run av train --output artifacts/perception --epochs 15
uv run av evaluate --checkpoint artifacts/perception/model.pt
uv run jupyter lab notebooks
```

CPU is the tested baseline. No accounts, external datasets, pretrained weights or GPU
are required. Each run needs a new output directory. Use only trusted checkpoints;
loading uses PyTorch's restricted `weights_only=True` mechanism.

## Notebook guide

All notebook names describe their subject without numeric prefixes. The main lab is
[perception_lab.ipynb](notebooks/perception_lab.ipynb). Each notebook runs independently
with package imports; no notebook depends on running another one first.

| Area | Notebooks |
| --- | --- |
| Perception | [Perception lab](notebooks/perception_lab.ipynb), [Synthetic data](notebooks/synthetic_data.ipynb), [Dataset manifests](notebooks/dataset_manifests.ipynb) |
| Learning | [Training experiments](notebooks/training_experiments.ipynb), [Augmentation and robustness](notebooks/augmentation_robustness.ipynb), [Uncertainty and calibration](notebooks/uncertainty_calibration.ipynb) |
| Tracking | [Object tracking](notebooks/object_tracking.ipynb) |
| Geometry | [Coordinate geometry](notebooks/coordinate_geometry.ipynb), [Road maps](notebooks/road_maps.ipynb), [Camera and depth](notebooks/camera_depth.ipynb) |
| Motion | [Vehicle dynamics](notebooks/vehicle_dynamics.ipynb), [Path planning](notebooks/path_planning.ipynb), [Speed profiles](notebooks/speed_profiles.ipynb) |
| Sensors | [Sensor mapping](notebooks/sensor_mapping.ipynb), [Sensor synchronization](notebooks/sensor_synchronization.ipynb), [Localization](notebooks/localization.ipynb) |
| Evaluation | [Closed-loop simulation](notebooks/closed_loop_simulation.ipynb), [Grid search](notebooks/grid_search.ipynb), [Benchmark analysis](notebooks/benchmark_analysis.ipynb), [Inference profiling](notebooks/inference_profiling.ipynb) |


## Commands

| Command | Result |
| --- | --- |
| `av train --output <new-directory>` | Best validation checkpoint and training JSON |
| `av evaluate --checkpoint <model.pt>` | Held-out synthetic segmentation metrics as JSON |
| `av simulate --output <new-directory>` | One closed-loop episode report |
| `av benchmark --output <new-directory> [--config <config.json>]` | Per-episode and aggregate reports |
| `av robustness --checkpoint <model.pt> --output <new-file.json>` | Fixed-seed corruption evaluation |
| `av inspect-dataset --manifest <manifest.json>` | Scene/split isolation and file validation summary |
| `av environment` | Runtime versions, without environment secrets |
| `av scenarios` | Available named synthetic scenarios |

`python -m autonomous_vehicles` also exposes these commands. See
[configuration examples](docs/development.md) and [contracts](docs/contracts.md).

## Verify

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv run python tools/execute_notebooks.py --output /tmp/av-executed-notebooks
uv run python tools/project_metrics.py --minimum 8000 --output /tmp/av-source-inventory.json
uv build
```

The line inventory counts Python files and notebook code cells, including normal
formatting/docstrings, and separately reports nonblank source and logical statements.
Notebook JSON, markdown, stored outputs, lockfiles and generated assets are excluded.
Line count is a requested inventory constraint; tests and experiments establish behavior.

Read [architecture](docs/architecture.md), [development](docs/development.md),
[version policy](docs/versions.md), and [reference analysis](docs/reference-analysis.md).
Repository licensing remains for the owner to select; dependencies retain their licenses.
