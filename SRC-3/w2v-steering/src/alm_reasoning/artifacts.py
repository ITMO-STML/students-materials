"""Atomic artifacts and validated prediction-chunk recovery."""

import hashlib
import json
from pathlib import Path
from typing import Any

from src.alm_reasoning.metrics import Cohorts, parse_answer
from src.steering.protocol import Trial


def write_json(path: Path, value: Any) -> None:
    """Atomically write a strict JSON artifact.

    Args:
        path: Destination artifact.
        value: JSON-compatible contents.
    """
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path: Path) -> str:
    """Hash a local artifact without loading it into memory.

    Args:
        path: Existing input artifact.

    Returns:
        SHA256 hexadecimal digest.
    """
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def restore_chunks(directory: Path, trials: list[Trial], cohorts: Cohorts) -> tuple[int, float]:
    """Validate ordered immutable predictions and rebuild pooled counters.

    Args:
        directory: Atomically saved chunk directory.
        trials: Full protocol in file order.
        cohorts: Empty accumulator to populate.

    Returns:
        Number of completed records and measured inference seconds.
    """
    offset, seconds = 0, 0.0
    for path in sorted(directory.glob("*.json")):
        chunk = json.loads(path.read_text())
        rows = chunk["predictions"]
        if chunk["start"] != offset or not rows or offset + len(rows) > len(trials):
            message = f"Noncontiguous or invalid chunk: {path}"
            raise ValueError(message)
        for trial, row in zip(trials[offset : offset + len(rows)], rows, strict=True):
            if row["utterance"] != trial.utterance or row["prediction"] != parse_answer(row["text"]):
                message = f"Prediction provenance mismatch: {path}"
                raise ValueError(message)
            cohorts.add(trial, row["prediction"])
        offset += len(rows)
        seconds += chunk["wall_seconds"]
    return offset, seconds
