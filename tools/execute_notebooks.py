"""Execute all notebooks in fresh kernels, saving results outside the source tree."""

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def execute_all(root: Path, output: Path, timeout: int = 180) -> list[Path]:
    if timeout < 1:
        raise ValueError("Timeout must be positive")
    notebooks = sorted((root / "notebooks").glob("*.ipynb"))
    if not notebooks:
        raise ValueError("No notebooks found")
    output.mkdir(parents=True, exist_ok=True)
    executed = []
    for path in notebooks:
        notebook = nbformat.read(path, as_version=4)
        client = NotebookClient(
            notebook,
            timeout=timeout,
            kernel_name="python3",
            resources={"metadata": {"path": str(root)}},
        )
        client.execute()
        destination = output / path.name
        nbformat.write(notebook, destination)
        executed.append(destination)
        print(f"Executed {path.name}", flush=True)
    return executed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    execute_all(root, args.output, args.timeout)


if __name__ == "__main__":
    main()
