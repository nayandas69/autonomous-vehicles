"""Atomic JSON artifacts and reproducible source/environment fingerprints."""

import hashlib
import json
import os
import platform
import tempfile
from dataclasses import asdict, is_dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np


def json_default(value):
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def atomic_json(path: Path, payload, overwrite: bool = False) -> None:
    """Write only complete JSON; reserve the destination without replacing a prior run."""
    serialized = json.dumps(payload, indent=2, default=json_default, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            # Hard link atomically fails if destination exists; both files share a filesystem.
            os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_fingerprint(root: Path) -> dict[str, str]:
    if not root.is_dir():
        raise ValueError("Source root must be a directory")
    return {
        str(path.relative_to(root)): file_sha256(path)
        for path in sorted(root.rglob("*.py"))
        if "__pycache__" not in path.parts
    }


def environment_metadata() -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.system(),
        "machine": platform.machine(),
        "torch": version("torch"),
        "numpy": version("numpy"),
        "package": version("autonomous-vehicles"),
    }


def stable_identifier(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=json_default, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]
