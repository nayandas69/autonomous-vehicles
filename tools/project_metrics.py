"""Count project Python and notebook code without counting JSON or lockfiles.

Physical source includes blank lines and docstrings. Nonblank source excludes
empty lines and standalone comments. Notebook markdown and stored outputs are
excluded. This report is an inventory, not a quality score.
"""

import argparse
import io
import json
import tokenize
from pathlib import Path


def source_counts(source: str) -> dict:
    lines = source.splitlines()
    nonblank = sum(bool(line.strip()) and not line.lstrip().startswith("#") for line in lines)
    tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    logical = sum(token.type == tokenize.NEWLINE for token in tokens)
    return {"physical": len(lines), "nonblank": nonblank, "logical_statements": logical}


def inventory(root: Path) -> dict:
    records = []
    for directory in ("src", "tests", "tools", "notebooks"):
        for path in sorted((root / directory).rglob("*")):
            if path.suffix == ".py":
                source = path.read_text()
            elif path.suffix == ".ipynb":
                notebook = json.loads(path.read_text())
                source = "\n".join(
                    "".join(cell["source"])
                    for cell in notebook["cells"]
                    if cell["cell_type"] == "code"
                )
            else:
                continue
            records.append({"path": str(path.relative_to(root)), **source_counts(source)})
    return {
        "definition": (
            "Python files and notebook code cells; excludes notebook JSON, "
            "markdown, outputs, lockfiles"
        ),
        "totals": {
            name: sum(record[name] for record in records)
            for name in ("physical", "nonblank", "logical_statements")
        },
        "files": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--minimum", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = inventory(args.root)
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(json.dumps(report["totals"], indent=2))
    if report["totals"]["physical"] < args.minimum:
        raise SystemExit(f"Source line count is below {args.minimum}")


if __name__ == "__main__":
    main()
