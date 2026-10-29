# Supported-version policy

Checked 2026-10-29 against the official Python lifecycle page and PyPI metadata.

| Component | Baseline | Policy |
| --- | --- | --- |
| Python | 3.14.x | Supported CPython minor; security support scheduled through October 2030 |
| PyTorch | 2.15.0 CPU | Pinned stable release; this project makes no vendor LTS claim |
| JupyterLab | 4.6.5 | Pinned stable release; no universal LTS guarantee |
| NumPy | 2.5.4 | Pinned stable release |
| OS | Ubuntu 26.04 LTS sandbox | Local validation environment |

Python, PyTorch, and Jupyter do not provide a common LTS designation. Therefore the
request for “all LTS” is implemented as an explicitly documented, supported and locked
baseline where LTS is not offered. Do not mistake pinned versions for indefinite support.
Python is constrained to the tested 3.14 minor. `uv.lock` records transitives; installation
and CI use `--locked`. The CPU index applies only to torch. A CUDA installation is a
separate untested profile, requiring a compatible driver and its own lock and validation.

Review dependencies monthly and when security advisories arrive. Upgrade in a branch,
regenerate the lock, execute tests and the notebook, and record compatibility before
merging. Revisit Python before its support window ends. Lockfile hashes provide artifact
integrity, not a vulnerability assessment.

Sources: [Python supported versions](https://devguide.python.org/versions/),
[PyTorch releases](https://pytorch.org/get-started/previous-versions/),
[PyTorch metadata](https://pypi.org/pypi/torch/json),
[JupyterLab metadata](https://pypi.org/pypi/jupyterlab/json).
