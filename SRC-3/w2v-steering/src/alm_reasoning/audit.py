"""Restricted-load metadata and recorded validation-overlap audit."""

import argparse
import json
from pathlib import Path

import torch

from src.alm_steering.run import write_json


def main() -> None:
    """Inspect an existing reasoning checkpoint without executing pickle code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, weights_only=True, mmap=True, map_location="cpu")
    state = checkpoint["model"]
    official = {line.split()[1] for line in args.protocol.read_text().splitlines() if line.strip()}
    recorded = set()
    trace_files = sorted((args.checkpoint.parent.parent / "logs").glob("samples_validation_epoch_*.jsonl"))
    for path in trace_files:
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("audio_id"):
                    recorded.add(str(row["audio_id"]))
    overlap = sorted(recorded & official)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output,
        {
            "checkpoint": str(args.checkpoint),
            "epoch": checkpoint.get("epoch"),
            "outer_keys": list(checkpoint),
            "state_keys": len(state),
            "state_metadata": {
                name: {"shape": list(value.shape), "dtype": str(value.dtype)} for name, value in state.items()
            },
            "checkpoint_config": checkpoint.get("config"),
            "validation_trace_files": [str(path) for path in trace_files],
            "unique_recorded_validation_ids": len(recorded),
            "official_eval_ids": len(official),
            "recorded_validation_overlap": len(overlap),
            "overlap_examples": overlap[:10],
            "lineage_limit": "Saved validation traces only; training-manifest disjointness remains unverified.",
        },
    )


if __name__ == "__main__":
    main()
