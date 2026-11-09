"""Audit SALMONN checkpoints without conflating token and class accuracy."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def read_log(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Parse a config followed by JSON-lines metrics.

    Args:
        path: Original SALMONN log file.

    Returns:
        Configuration and validation records in epoch order.
    """
    text = path.read_text()
    config, end = json.JSONDecoder().raw_decode(text.lstrip())
    records = [json.loads(line) for line in text.lstrip()[end:].splitlines() if line.startswith("{")]
    return config, [row for row in records if "valid_agg_metrics" in row]


def inventory(root: Path) -> dict[str, Any]:
    """Rank only explicit classification metrics from available best checkpoints.

    Args:
        root: Directory containing date-named SALMONN runs.

    Returns:
        Separate metric records, selected candidate, and comparability caveats.
    """
    rows = []
    for log in sorted(root.glob("202*/log.txt")):
        config, metrics = read_log(log)
        if not metrics:
            continue
        checkpoint = log.parent / "checkpoint_best.pth"
        best_epoch = int(metrics[-1]["valid_best_epoch"])
        best = metrics[best_epoch]
        rows.append(
            {
                "run": log.parent.name,
                "checkpoint": str(checkpoint),
                "available": checkpoint.is_file(),
                "best_epoch": best_epoch,
                "metrics": best,
                "config": config,
                "metric_kind": "classification" if "valid_classification_acc" in best else "token_or_unspecified",
            },
        )
    eligible = [row for row in rows if row["available"] and row["metric_kind"] == "classification"]
    if not eligible:
        msg = "No available checkpoint has explicit validation classification accuracy"
        raise ValueError(msg)
    selected = max(eligible, key=lambda row: row["metrics"]["valid_classification_acc"])
    with Path(selected["checkpoint"]).open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {
        "runs": rows,
        "selected": selected,
        "checkpoint_sha256": digest,
        "caveat": "Classification accuracy on a small legacy validation set; not EER or reasoning accuracy. "
        "Token/unspecified aggregate metrics are not comparable. Training-manifest lineage needs separate audit.",
    }


def main() -> None:
    """Save an inventory and restricted-load inspection of the selected checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = inventory(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    import torch

    checkpoint = torch.load(report["selected"]["checkpoint"], map_location="cpu", weights_only=True, mmap=True)
    state = checkpoint["model"]
    report["tensor_shapes"] = {name: list(value.shape) for name, value in state.items()}
    report["saved_epoch"] = checkpoint.get("epoch")
    report["checkpoint_fields"] = list(checkpoint)
    assert int(checkpoint["epoch"]) == report["selected"]["best_epoch"]
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
