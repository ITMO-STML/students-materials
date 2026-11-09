"""Add mean/sample-std reports to saved benchmarks without rerunning inference."""

import argparse
import csv
import hashlib
import json
import logging
from pathlib import Path

from src.steering.compute import summarize_rounds
from src.steering.run import write_json


def summarize_saved(path: Path) -> Path:
    """Write derived summary files while preserving the original run artifacts.

    Args:
        path: Existing compute.json containing the individual benchmark rounds.

    Returns:
        Path to the derived JSON report, with input and summarizer provenance.
    """
    original = json.loads(path.read_text())
    summary = summarize_rounds(
        original["inference"]["samples"],
        int(original["clips_per_repeat"]),
        int(original["samples_per_clip"]) / 16000,
    )
    destination = path.with_name("compute_mean_std.json")
    write_json(
        destination,
        {
            "source": str(path),
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "summarizer_sha256": {
                str(source): hashlib.sha256(source.read_bytes()).hexdigest()
                for source in (Path(__file__), Path(__file__).with_name("compute.py"))
            },
            "interpretation": (
                "Arithmetic mean +/- sample std (ddof=1) across measured benchmark rounds, "
                "not clips, confidence intervals, or adaptation seeds. Overheads are paired by round. "
                "Original median summaries and raw measurements remain unchanged."
            ),
            "unreplicated": (
                "EER and adaptation-stage times have one run each: seed-to-seed std is unavailable. "
                "Different layers/stages are not repetitions of the same workload."
            ),
            "summary": summary,
        },
    )
    with destination.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", *next(iter(summary.values())).keys()])
        writer.writeheader()
        for method, values in summary.items():
            writer.writerow({"method": method, **values})
    return destination


def main() -> None:
    """Parse a saved compute report and log its derived summary destination."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("compute_json", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    logging.getLogger(__name__).info("Saved mean/std summary: %s", summarize_saved(args.compute_json))


if __name__ == "__main__":
    main()
