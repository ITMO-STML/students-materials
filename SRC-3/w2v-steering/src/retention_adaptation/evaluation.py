"""Frozen retention-adapted checkpoints evaluated through the shared eval.py API."""

import csv
import hashlib
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_file
from torch.utils.data import DataLoader

from src.steering.evaluation import attack_report
from src.steering.external_protocol import ExternalAudio, ExternalRecording, inventory
from src.steering.finetune import build_model
from src.steering.protocol import Trial, read_trials
from src.steering.run import write_json

from .protocol import digest


def evaluation_rows(cfg: DictConfig, adaptation: Path) -> list[ExternalRecording]:
    """Load the held-out cohort, full ASVspoof eval, or an audited external manifest.

    Args:
        cfg: Composed evaluation settings.
        adaptation: Completed fitting run.

    Returns:
        Unique, ordered recording metadata with existing preprocessing semantics.
    """
    cohort = str(cfg.evaluation.cohort)
    if cohort == "external":
        reference = Path(cfg.evaluation.manifest_run)
        if json.loads((reference / "report.json").read_text())["status"] != "complete":
            message = "External manifest must come from a completed evaluation"
            raise ValueError(message)
        manifest = json.loads((reference / "manifest.json").read_text())
        names = {field.name for field in fields(ExternalRecording)}
        rows = [ExternalRecording(**{k: v for k, v in row.items() if k in names}) for row in manifest["recordings"]]
        if inventory(rows) != manifest:
            message = "External audio inventory changed since the audited evaluation"
            raise ValueError(message)
        return rows
    if cohort == "asvspoof5_full":
        trials = read_trials(Path(cfg.data.root), "eval")
    elif cohort in {"test", "dev"}:
        raw = json.loads((adaptation / "manifest.json").read_text())
        trials = [Trial(**row) for row in raw[cohort]]
    else:
        message = "Unknown retention evaluation cohort"
        raise ValueError(message)
    return [
        ExternalRecording(
            t.utterance,
            str(Path(t.audio_dir) / f"{t.utterance}.flac"),
            t.label,
            t.attack if t.label else "bonafide",
            t.speaker,
            source=t.source,
        )
        for t in trials
    ]


def score(
    model: torch.nn.Module,
    rows: list[ExternalRecording],
    cfg: DictConfig,
    output: Path,
    method: str,
) -> np.ndarray:
    """Score deterministic batches, resuming only contiguous validated chunks.

    Args:
        model: Frozen detector with the requested checkpoint loaded.
        rows: Fixed evaluation inventory.
        cfg: Runtime and preprocessing settings.
        output: Run directory whose identity has already been checked.
        method: baseline or adapted.

    Returns:
        Complete aligned genuine-positive scores.
    """
    directory = output / "chunks" / method
    directory.mkdir(parents=True, exist_ok=True)
    values = []
    start = 0
    for path in sorted(directory.glob("*.npz")):
        with np.load(path, allow_pickle=False) as chunk:
            indices, saved = chunk["indices"], chunk["scores"]
            if saved.ndim != 1 or not len(saved) or not np.array_equal(indices, np.arange(start, start + len(saved))):
                message = "Noncontiguous or malformed score chunks"
                raise ValueError(message)
            if not np.isfinite(saved).all() or (saved < 0).any() or (saved > 1).any():
                message = "Invalid saved probabilities"
                raise ValueError(message)
            values.append(saved)
            start += len(saved)
    if start > len(rows):
        message = "More saved scores than recordings"
        raise ValueError(message)
    loader = DataLoader(
        ExternalAudio(rows, int(cfg.data.num_samples), start),
        batch_size=int(cfg.steering.batch_size),
        num_workers=0,
        shuffle=False,
    )
    device = next(model.parameters()).device
    model.eval()
    with torch.inference_mode():
        for indices, waves in loader:
            inputs = waves.to(device)
            logits = model(input_values=inputs, attention_mask=torch.ones_like(inputs, dtype=torch.long))["logits"]
            probabilities = logits.float().softmax(-1)[:, 0].cpu().numpy()
            if not np.isfinite(probabilities).all():
                message = "Nonfinite detector scores"
                raise ValueError(message)
            path = directory / f"{int(indices[0]):09d}.npz"
            temporary = path.with_suffix(".tmp")
            with temporary.open("wb") as handle:
                np.savez(handle, indices=indices.numpy(), scores=probabilities)
            temporary.replace(path)
            values.append(probabilities)
            start += len(probabilities)
            write_json(
                output / "progress.json",
                {"status": "scoring", "method": method, "processed": start, "total": len(rows)},
            )
    return np.concatenate(values)


def validate_identity(output: Path, identity: dict[str, Any]) -> None:
    """Reject mixed checkpoints or manifests before resuming saved scores.

    Args:
        output: Evaluation output directory.
        identity: Current immutable inference identity.
    """
    provenance = output / "provenance.json"
    if provenance.exists() and json.loads(provenance.read_text()) != identity:
        message = "Evaluation identity changed; use a new output directory"
        raise ValueError(message)
    if not provenance.exists() and (output / "chunks").exists():
        message = "Cannot resume scores without provenance"
        raise ValueError(message)
    write_json(provenance, identity)


def run(cfg: DictConfig, output: Path) -> None:
    """Evaluate selected frozen weights; never fit or recalibrate on test data.

    Args:
        cfg: Shared eval.py configuration with evaluation=retention_a21.
        output: Exclusively locked evaluation directory.
    """
    adaptation = Path(cfg.evaluation.adaptation_run)
    selected = json.loads((adaptation / "selection.json").read_text())
    if selected["status"] != "complete" or cfg.evaluation.selection not in {"guarded", "best_a21"}:
        message = "Require completed training and a frozen checkpoint selection"
        raise ValueError(message)
    plan = json.loads((adaptation / "plan.json").read_text())
    fitting = OmegaConf.create(plan["config"])
    fitting.steering.device = cfg.steering.device
    if int(cfg.data.num_samples) != int(fitting.data.num_samples):
        message = "Evaluation waveform length differs from fitting"
        raise ValueError(message)
    paths = {
        "baseline": Path(selected["baseline_checkpoint"]),
        "adapted": Path(selected[f"{cfg.evaluation.selection}_checkpoint"]),
    }
    hashes = {name: digest(path) for name, path in paths.items()}
    if (
        hashes["baseline"] != selected["checkpoint_sha256"]["baseline"]
        or hashes["adapted"] != selected["checkpoint_sha256"][str(cfg.evaluation.selection)]
    ):
        message = "Selected checkpoint changed after fitting"
        raise ValueError(message)
    rows = evaluation_rows(cfg, adaptation)
    if len({r.key for r in rows}) != len(rows) or {r.label for r in rows} != {0, 1}:
        message = "Require unique recordings and both classes"
        raise ValueError(message)
    manifest = inventory(rows)
    sources = [
        *Path(__file__).parent.glob("*.py"),
        *Path("src/model").glob("*.py"),
        *Path("src/metrics").glob("*.py"),
        *Path("src/data").glob("*.py"),
        Path("src/steering/evaluation.py"),
        Path("src/steering/external_protocol.py"),
        Path("src/steering/finetune.py"),
    ]
    identity: dict[str, Any] = {
        "manifest_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
        "checkpoints": hashes,
        "selection_sha256": digest(adaptation / "selection.json"),
        "num_samples": int(cfg.data.num_samples),
        "batch_size": int(cfg.steering.batch_size),
        "threshold": selected["threshold"],
        "torch": str(torch.__version__),
        "precision": "FP32",
        "sources": {str(p): digest(p) for p in sorted(sources)},
    }
    validate_identity(output, identity)
    write_json(output / "manifest.json", manifest)
    if cfg.steering.prepare_only:
        return
    model = build_model(fitting)
    result = {}
    for method, path in paths.items():
        model.load_state_dict(load_file(path), strict=True)
        values = score(model, rows, cfg, output, method)
        result[method] = attack_report([r.trial() for r in rows], values, float(selected["threshold"]))
    write_json(
        output / "report.json",
        {
            "status": "complete",
            "cohort": str(cfg.evaluation.cohort),
            "n": len(rows),
            "threshold": selected["threshold"],
            "results": result,
            "note": (
                "Unique-recording pooled and per-generator/datasource metrics. "
                "Full ASVspoof eval includes adaptation support/tune and is not fully held out."
            ),
        },
    )
    with (output / "comparison.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "group", "eer_percent", "min_dcf", "fixed_dcf", "bonafide_rejection_percent"])
        for method, groups in result.items():
            for group, row in groups.items():
                writer.writerow(
                    [
                        method,
                        group,
                        100 * row["eer"],
                        row["min_dcf"],
                        row["fixed_dcf"],
                        100 * row["bonafide_rejection"],
                    ],
                )
    write_json(output / "progress.json", {"status": "complete", "processed": len(rows), "total": len(rows)})
