"""Classification-only reporting from saved ALM scores, without inference."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.alm_steering.classification import classification_rank, classification_report, retention_allowed
from src.steering.protocol import Trial


def write_json(path: Path, value: Any) -> None:
    """Persist strict JSON atomically.

    Args:
        path: New run-owned destination.
        value: Serializable artifact.
    """
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def metrics(path: Path, trials: list[Trial]) -> dict[str, dict[str, float | int]]:
    """Read saved scores and aggregate the requested classification metrics.

    Args:
        path: Existing score artifact.
        trials: Exact corresponding manifest records.

    Returns:
        Pooled and per-attack classification metrics.
    """
    scores = np.asarray(json.loads(path.read_text())["scores"])
    return classification_report(trials, scores)


def rank(row: dict[str, Any]) -> tuple[float, ...]:
    """Build a tuning-only rank with deterministic steering tie-breaks.

    Args:
        row: Candidate with tuning metrics, strength and layer.

    Returns:
        Ascending-sort selection key.
    """
    return (*classification_rank(row["tune"]), abs(row["strength"]), row["layer"])


def reselect(reference: Path, trials: dict[str, list[Trial]]) -> dict[str, Any]:
    """Reselect from saved tuning scores, explicitly flagging missing dev data.

    Args:
        reference: Original steering pilot.
        trials: Exact manifest groups.

    Returns:
        New selection plus missing inference requirements, never silently dropped.
    """
    directory = reference / "scores"
    baseline = {name: metrics(directory / f"baseline_{name}.json", trials[name])["pooled"] for name in ("tune", "dev")}
    old = json.loads((reference / "selection.json").read_text())["selected"]
    selected = {
        method: {
            "family": row["family"],
            "scope": row["scope"],
            "layer": 8,
            "strength": 0.0,
            "tune": baseline["tune"],
            "dev": baseline["dev"],
        }
        for method, row in old.items()
    }
    pending: list[tuple[str, dict[str, Any]]] = []
    candidates: list[dict[str, Any]] = []
    for row in json.loads((reference / "candidates.json").read_text()):
        setting = {key: row[key] for key in ("family", "scope", "layer", "strength")}
        method = f"{setting['family']}__{setting['scope']}"
        slug = f"{method}_L{setting['layer']}_a{setting['strength']}"
        tune = metrics(directory / f"{slug}_tune.json", trials["tune"])["pooled"]
        dev_path = directory / f"{slug}_dev.json"
        dev = metrics(dev_path, trials["dev"])["pooled"] if dev_path.exists() else None
        tune_allowed = tune["bonafide_recall"] >= baseline["tune"]["bonafide_recall"] - 0.05
        candidate = {**setting, "tune": tune, "dev": dev}
        candidate["admissible"] = tune_allowed and dev is not None and retention_allowed(dev, baseline["dev"])
        candidates.append(candidate)
        if tune_allowed and dev is None:
            pending.append((method, candidate))
        if candidate["admissible"] and rank(candidate) < rank(selected[method]):
            selected[method] = candidate
    unresolved = [{"method": method, "setting": row} for method, row in pending if rank(row) < rank(selected[method])]
    results: dict[str, dict[str, dict[str, float | int]]] = {}
    missing_test: list[str] = []
    for method, setting in selected.items():
        filename = directory / f"{method}_test.json"
        if setting["strength"] == 0:
            filename = directory / "baseline_test.json"
        elif any(setting[key] != old[method][key] for key in ("layer", "strength", "scope")):
            missing_test.append(method)
            continue
        results[method] = metrics(filename, trials["test"])
    return {
        "status": "needs_development_inference" if unresolved else "selection_complete",
        "selected": selected,
        "candidates": candidates,
        "missing_development": unresolved,
        "missing_test_methods": missing_test,
        "available_test_results": results if not unresolved else {},
        "note": "Tuning/development-only reselection; historical pilot test results were previously inspected.",
    }


def finetune_report(reference: Path, trials: dict[str, list[Trial]]) -> dict[str, Any]:
    """Report completed epochs without calling interrupted training complete.

    Args:
        reference: Interrupted LoRA run.
        trials: Exact manifest groups.

    Returns:
        Recomputed validation metrics and provisional best completed epoch.
    """
    history: list[dict[str, Any]] = []
    for epoch in range(5):
        prefix = "baseline" if epoch == 0 else f"epoch_{epoch}"
        values = {
            name: metrics(reference / f"{prefix}_{name}.json", trials[name])["pooled"] for name in ("tune", "dev")
        }
        history.append({"epoch": epoch, "metrics": values})
    baseline = history[0]["metrics"]
    for row in history:
        row["admissible"] = row["metrics"]["tune"]["bonafide_recall"] >= baseline["tune"][
            "bonafide_recall"
        ] - 0.05 and retention_allowed(row["metrics"]["dev"], baseline["dev"])
    best = min(
        (row for row in history if row["admissible"]),
        key=lambda row: classification_rank(row["metrics"]["tune"]),
    )
    return {
        "training_status": "interrupted during epoch 5; four complete epochs only",
        "history": history,
        "best_completed_epoch": best["epoch"],
        "test_status": "not_evaluated",
    }


def main() -> None:
    """Write fresh classification artifacts while preserving original outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--finetune", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    raw = json.loads((args.reference / "manifest.json").read_text())
    trials = {name: [Trial(**row) for row in rows] for name, rows in raw.items()}
    original = json.loads((args.reference / "report.json").read_text())
    reports = {}
    for method in original["results"]:
        filename = "prompting_test.json" if method == "prompting_only" else f"{method}_test.json"
        reports[method] = metrics(args.reference / "scores" / filename, trials["test"])
    write_json(
        args.output / "report.json",
        {
            "cohort": "4352-recording historical pilot, NOT full evaluation",
            "positive_class": "spoof",
            "decision_rule": "logP(bonafide)-logP(spoof) < 0 predicts spoof; ties bonafide",
            "method_selection": "historical fixed settings; this table only changes evaluation metrics",
            "results": reports,
        },
    )
    with (args.output / "comparison.csv").open("w") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "method",
                "attack",
                "accuracy_percent",
                "precision_percent",
                "recall_percent",
                "f1_percent",
                "tp",
                "tn",
                "fp",
                "fn",
                "n_bonafide",
                "n_spoof",
            ],
        )
        for method, groups in reports.items():
            for attack, row in groups.items():
                writer.writerow(
                    [
                        method,
                        attack,
                        *[100 * row[key] for key in ("accuracy", "precision", "recall", "f1")],
                        *[row[key] for key in ("tp", "tn", "fp", "fn", "n_bonafide", "n_spoof")],
                    ],
                )
    write_json(args.output / "classification_selection.json", reselect(args.reference, trials))
    write_json(args.output / "finetune_validation.json", finetune_report(args.finetune, trials))
    files = [args.reference / name for name in ("manifest.json", "selection.json", "candidates.json", "report.json")]
    files += list((args.reference / "scores").glob("*.json"))
    files += list(args.finetune.glob("*_tune.json")) + list(args.finetune.glob("*_dev.json"))
    files += [Path(__file__), Path("src/alm_steering/classification.py")]
    write_json(
        args.output / "provenance.json",
        {"sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}},
    )
    write_json(args.output / "progress.json", {"stage": "complete", "status": "complete", "inference_rerun": False})


if __name__ == "__main__":
    main()
