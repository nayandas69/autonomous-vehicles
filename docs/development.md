# Development and experiments

## Environment

Use Python 3.14 and the checked-in uv lock. `uv sync --locked --all-extras` installs the
CPU training environment, development checks and notebook tools. Ordinary startup
only needs `uv run jupyter lab notebooks`; it does not require reinstalling dependencies.
No external API or local emulator is used. See `versions.md` for support policy.

## Batch configuration

Save the following as a JSON file, then pass `--config` to `av benchmark`:

```json
{
  "scenarios": ["lane_following", "winding_road", "stopped_vehicle"],
  "controllers": ["pursuit", "stanley"],
  "safety": true,
  "repetitions": 1
}
```

Unknown keys, unknown scenarios, duplicate names, and invalid types are errors.
Repetitions exercise reproducibility; scenario diversity or perturbed configurations
are needed for independent evidence. Output includes per-record JSON and a complete
report with configuration, experiment identifier and runtime versions.

## Notebook workflow

Each notebook starts from package imports and constructs its own fixtures. File-writing
examples use temporary directories. Committed notebooks have no saved outputs or
execution counts. They can run from Jupyter opened at the repository root or the
notebooks directory because the package is installed in the environment.

Execute the entire collection with:

```bash
uv run python tools/execute_notebooks.py --output /tmp/av-executed-notebooks
```

Each notebook receives a fresh kernel. Outputs are written separately, leaving source
notebooks clean. The timeout applies per cell. Optional matplotlib dependencies live
in the `notebooks` extra; importing the core package needs no display backend.

## Local dataset adapter

The `ManifestDataset` loader validates local NumPy files. To create a synthetic example:

```python
from pathlib import Path
from autonomous_vehicles.perception.export import export_scenes, verify_export
from autonomous_vehicles.perception.dataset import ManifestDataset

manifest = export_scenes(Path("artifacts/data-example"), samples_per_split=8)
dataset = ManifestDataset(manifest, "train")
print(verify_export(manifest.parent))
```

The exporter writes file hashes and generator configuration. Verification detects
changed files against that local baseline; it does not authenticate a dataset publisher.
For real datasets, preserve scene identity, map class labels explicitly and provide
an appropriate dataset card before training.

## Testing and source inventory

Tests use analytical geometry, exact metric cases, physical bounds, malformed inputs,
transactional writes, split leakage checks, RNG restoration and installed CLI commands.
No tests assert a particular wall-clock speed in the shared sandbox. Profiling reports
hardware-dependent observations only.

`tools/project_metrics.py` inventories Python files and notebook code-cell source.
Physical source includes regular blank lines and docstrings. Nonblank source excludes
empty lines and standalone comments; logical statements count Python NEWLINE tokens.
These are different metrics. The requested minimum is enforced using physical source
without counting notebook JSON, markdown, outputs, lockfiles or documentation. Source
size is not evidence of correctness; no artificial repeated declarations were added.

CI runs lint/format, the full test suite, all notebooks, the source inventory gate,
and package builds. Local validation is recorded outside Git in workspace context.
Remote CI results become available after publishing the branch.

## Version changes

The initial `data`, `model`, `engine` and `metrics` APIs remain available. The notebook
path intentionally changes from its old numeric name to `notebooks/perception_lab.ipynb`,
as requested. README and CI references use the new name. New functionality is exported
from dedicated modules rather than importing every optional feature at package import.
