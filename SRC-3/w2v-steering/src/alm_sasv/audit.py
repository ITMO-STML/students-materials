"""Inspect the user-specified SASV checkpoint and native pair manifests."""

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from src.alm_reasoning.artifacts import write_json


def main() -> None:
    """Save restricted checkpoint metadata and pair-manifest inventory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pairs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, weights_only=True, mmap=True, map_location="cpu")
    state = checkpoint["model"]
    cfg = checkpoint.get("config", {})
    metadata = {
        "checkpoint": str(args.checkpoint),
        "epoch": checkpoint.get("epoch"),
        "state_keys": len(state),
        "outer_keys": list(checkpoint),
        "embedded_config_keys": list(cfg),
        "embedded_generation": cfg.get("generation"),
        "embedded_task_type": cfg.get("task_type"),
        "tensor_metadata": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in state.items()},
        "pair_manifests": {},
    }
    for name in ("sasv_asv5_dev_pairs_balanced.json", "sasv_asv5_eval_pairs_20K.json", "sasv_asv5_eval_pairs.json"):
        path = args.pairs_root / name
        rows = json.loads(path.read_text())
        first_paths = [
            Path(a.get("original_path") or a["path"])
            for row in rows[:2048]
            for key in ("reference_audios", "query_audios")
            for a in row[key]
        ]
        metadata["pair_manifests"][name] = {
            "records": len(rows),
            "labels": dict(Counter(row["gt"] for row in rows)),
            "first_2048_labels": dict(Counter(row["gt"] for row in rows[:2048])),
            "record_fields": sorted(rows[0]),
            "reference_counts": dict(Counter(len(row["reference_audios"]) for row in rows)),
            "query_counts": dict(Counter(len(row["query_audios"]) for row in rows)),
            "first_2048_missing_audio": sum(not p.is_file() for p in first_paths),
            "unique_queries": len(
                {Path(a.get("original_path") or a["path"]).stem for row in rows for a in row["query_audios"]},
            ),
        }
        del rows
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, metadata)


if __name__ == "__main__":
    main()
