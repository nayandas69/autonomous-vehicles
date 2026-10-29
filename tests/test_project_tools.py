"""Verify source inventory excludes generated JSON and notebook outputs."""

import importlib.util
import json
import tokenize
from pathlib import Path

import pytest


def metrics_module():
    path = Path(__file__).resolve().parents[1] / "tools/project_metrics.py"
    specification = importlib.util.spec_from_file_location("project_metrics", path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_inventory_counts_only_python_and_notebook_code(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    (source / "example.py").write_text("# comment\nx = 1\n\ny = 2\n")
    (tmp_path / "uv.lock").write_text("not source\n" * 100)
    notebooks = tmp_path / "notebooks"
    notebooks.mkdir()
    notebook = {
        "cells": [
            {"cell_type": "markdown", "source": ["documentation\n" * 100]},
            {
                "cell_type": "code",
                "source": ["z = 3\n"],
                "outputs": [{"text": "large stored output\n" * 100}],
            },
        ]
    }
    (notebooks / "example.ipynb").write_text(json.dumps(notebook, indent=2))
    report = metrics_module().inventory(tmp_path)
    assert report["totals"] == {"physical": 5, "nonblank": 3, "logical_statements": 3}
    assert len(report["files"]) == 2


def test_source_counts_distinguish_physical_and_logical_lines():
    source = "value = (\n    1 +\n    2\n)\n# explanation\n\n"
    result = metrics_module().source_counts(source)
    assert result["physical"] == 6
    assert result["nonblank"] == 4
    assert result["logical_statements"] == 1


def test_notebook_source_is_parseable_python():
    root = Path(__file__).resolve().parents[1]
    notebooks = list((root / "notebooks").glob("*.ipynb"))
    assert notebooks
    for path in notebooks:
        notebook = json.loads(path.read_text())
        assert not path.stem[0].isdigit(), path.name
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), str(path), "exec")
                assert cell["outputs"] == [], path.name
                assert cell["execution_count"] is None, path.name


def test_inventory_reports_bad_python_instead_of_silently_counting_it():
    with pytest.raises(tokenize.TokenError):
        metrics_module().source_counts("value = (\n")
