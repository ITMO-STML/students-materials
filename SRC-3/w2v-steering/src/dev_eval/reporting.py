"""Refresh the report table from completed evaluations without provisional metrics."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.steering.external_eval import external_results
from src.steering.external_protocol import ExternalRecording
from src.steering.run import write_json

METHODS = {
    "Baseline": "baseline",
    "Dense steering, strength 0": "dense",
    "SAE + steering": "sae",
    "SAE reconstruction only": "reconstruction",
    "Shuffled-feature SAE steering": "random_sae",
    "A21 fine-tuning, epoch 5": "finetune_a21",
    "Guarded fine-tuning, epoch 0": "finetune_guarded",
    "MidSteer-inspired affine": "midsteer_affine",
    "Anchored/gated transport": "midsteer_anchored_gated",
    "OY dense, nonzero diagnostic": "speaker_dense_nonzero",
    "OY SAE, nonzero diagnostic": "speaker_sae_nonzero",
    "OY affine, nonzero diagnostic": "speaker_affine_nonzero",
    "OY gated, nonzero diagnostic": "speaker_gated_nonzero",
    "CLEAR projected, best A21 (diagnostic)": "clear_projected_best_a21",
    "CLEAR projected, guarded": "clear_projected_guarded",
}
WILD_COLUMNS = {
    "WildSpoof / ASVspoof 5": "datasource_asvspoof5",
    "WildSpoof / SpoofCeleb": "datasource_spoofceleb",
    "WildSpoof / SASV 2020": "datasource_sasv2020",
}
DEV_COLUMNS = {"Dev A09": "A09", "Dev A10": "A10", "ASVspoof 5 dev pooled": "pooled"}


def completed(path: Path) -> dict[str, Any] | None:
    """Read a report only if its writer marked it complete.

    Args:
        path: Report JSON path, possibly absent during a running evaluation.

    Returns:
        Completed report or None.
    """
    if not path.exists():
        return None
    result = json.loads(path.read_text())
    return result if result.get("status") == "complete" else None


def clear_wild_cohorts(root: Path) -> dict[str, Any] | None:
    """Derive datasource-only metrics from completed new checkpoint scores.

    Args:
        root: Completed retention evaluator output with per-method chunks.

    Returns:
        Unchanged WildSpoof cohort metrics or None while inference is incomplete.
    """
    report = completed(root / "report.json")
    if report is None:
        return None
    manifest = json.loads((root / "manifest.json").read_text())
    names = set(ExternalRecording.__dataclass_fields__)
    rows = [ExternalRecording(**{k: v for k, v in row.items() if k in names}) for row in manifest["recordings"]]
    scores = {}
    for method in ("baseline", "adapted"):
        parts = []
        offset = 0
        for path in sorted((root / "chunks" / method).glob("*.npz")):
            with np.load(path, allow_pickle=False) as chunk:
                probabilities = chunk["scores"]
                if not np.array_equal(chunk["indices"], np.arange(offset, offset + len(probabilities))):
                    message = "CLEAR WildSpoof scores have a gap or overlap"
                    raise ValueError(message)
                parts.append(probabilities)
                offset += len(probabilities)
        if offset != len(rows):
            message = "CLEAR WildSpoof report is not backed by complete scores"
            raise ValueError(message)
        scores[method] = np.concatenate(parts)
    result = external_results(rows, scores, float(report["threshold"]), "wildspoof_2026")
    write_json(root / "datasource_report.json", {"status": "complete", "results": result})
    return result


def replace_table(text: str, values: dict[str, dict[str, str]]) -> str:
    """Extend the main w2v table by column name, preserving unrelated report text.

    Args:
        text: Current research report.
        values: Method labels mapped to available column strings.

    Returns:
        Updated report; repeated refreshes produce the same table.
    """
    section = text.index("### 4.1. Wav2vec2")
    start = text.index("| Method |", section)
    end = text.index("\n\n", start)
    lines = text[start:end].splitlines()
    header = [cell.strip() for cell in lines[0].strip("|").split("|")]
    existing = {}
    for line in lines[2:]:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        existing[cells[0]] = dict(zip(header, cells, strict=True))
    for column in [*WILD_COLUMNS, *DEV_COLUMNS]:
        if column not in header:
            header.append(column)
    output = ["| " + " | ".join(header) + " |", "| --- | " + " | ".join(["---:"] * (len(header) - 1)) + " |"]
    for label in METHODS:
        row = existing.get(label, {"Method": label})
        row.update(values.get(label, {}))
        output.append("| " + " | ".join(row.get(column, "—") for column in header) + " |")
    return text[:start] + "\n".join(output) + text[end:]


def update(report: Path, dev_run: Path, fit_run: Path, wild_best: Path, wild_guarded: Path) -> None:
    """Populate observed results and keep unfinished cohorts explicitly pending.

    Args:
        report: Destination Markdown research report.
        dev_run: Full development evaluator output.
        fit_run: CLEAR fitting run, possibly incomplete.
        wild_best: CLEAR best-A21 WildSpoof evaluation.
        wild_guarded: Distinct guarded-checkpoint WildSpoof evaluation if needed.
    """
    wild = json.loads(Path("logs/runs/2026-10-18/w2v_oy_wildspoof_v3/report.json").read_text())["results"]
    dev = completed(dev_run / "report.json")
    selected = completed(fit_run / "selection.json")
    best = clear_wild_cohorts(wild_best)
    guarded = clear_wild_cohorts(wild_guarded)
    clear = {"clear_projected_best_a21": (best, "adapted"), "clear_projected_guarded": (guarded, "adapted")}
    if selected is not None:
        hashes = selected["checkpoint_sha256"]
        if hashes["guarded"] == hashes["baseline"]:
            clear["clear_projected_guarded"] = (wild, "baseline")
        elif hashes["guarded"] == hashes["best_a21"]:
            clear["clear_projected_guarded"] = (best, "adapted")
    values = {}
    for label, method in METHODS.items():
        row = dict.fromkeys([*WILD_COLUMNS, *DEV_COLUMNS], "pending")
        cohorts, key = clear.get(method, (wild, method))
        if cohorts is not None:
            for column, cohort in WILD_COLUMNS.items():
                row[column] = f"{100 * cohorts[cohort]['metrics'][key]['pooled']['eer']:.2f}"
            row["WildSpoof 2026"] = f"{100 * cohorts['all_unique_queries']['metrics'][key]['pooled']['eer']:.2f}"
        if dev is not None and method in dev["results"]:
            for column, group in DEV_COLUMNS.items():
                row[column] = f"{100 * dev['results'][method][group]['eer']:.2f}"
        values[label] = row
    updated = replace_table(report.read_text(), values)
    temporary = report.with_suffix(".tmp")
    temporary.write_text(updated)
    temporary.replace(report)


def main() -> None:
    """Refresh available results after any completed queued stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path("docs/REPORT.md"))
    parser.add_argument("--dev-run", type=Path, default=Path("logs/runs/2026-11-06/w2v_full_dev_clear_v1"))
    parser.add_argument("--fit-run", type=Path, default=Path("logs/runs/2026-11-06/clear_projected_fit_v1"))
    parser.add_argument("--wild-best", type=Path, default=Path("logs/runs/2026-11-06/clear_wildspoof_best_v1"))
    parser.add_argument("--wild-guarded", type=Path, default=Path("logs/runs/2026-11-06/clear_wildspoof_guarded_v1"))
    args = parser.parse_args()
    update(args.report, args.dev_run, args.fit_run, args.wild_best, args.wild_guarded)


if __name__ == "__main__":
    main()
