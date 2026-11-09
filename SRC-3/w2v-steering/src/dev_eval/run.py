"""Evaluate full development cohorts through eval.py, without reselection."""

import csv
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_file

from src.speaker_steering.evaluation import METHODS as OY_METHODS
from src.speaker_steering.evaluation import fitted_artifacts, inference_jobs
from src.steering.external_eval import digest, provenance, stream, validate_identity
from src.steering.external_protocol import ExternalAudio, ExternalRecording, inventory
from src.steering.finetune import build_model
from src.steering.full_eval import Scorer, build_scorers, score_batch, smoke_check
from src.steering.protocol import read_trials
from src.steering.run import write_json
from src.voxenes_eval.run import metrics, smoke_samples

log = logging.getLogger(__name__)
CLEAR_METHODS = ("clear_projected_best_a21", "clear_projected_guarded")


def development_rows(cfg: DictConfig) -> list[ExternalRecording]:
    """Validate and retain every recording from the official development protocol.

    Args:
        cfg: Configuration with expected full-population counts.

    Returns:
        Ordered genuine and A09-A16 records, without sampling or deduplication.
    """
    trials = read_trials(Path(cfg.data.root), "dev")
    expected = {"bonafide": int(cfg.steering.expected_bonafide)}
    expected.update({f"A{i:02d}": int(cfg.steering.expected_per_attack) for i in range(9, 17)})
    if Counter(t.attack for t in trials) != expected or len({t.utterance for t in trials}) != len(trials):
        message = "Development protocol has unexpected counts, attacks or duplicate recordings"
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


def clear_artifacts(cfg: DictConfig, baseline_hash: str, threshold: float) -> tuple[dict[str, Path], dict[str, str]]:
    """Verify completed CLEAR selection and return immutable model artifacts.

    Args:
        cfg: Evaluation settings with an optional clear_run directory.
        baseline_hash: Original comparator checkpoint hash.
        threshold: Frozen baseline development threshold.

    Returns:
        Named checkpoint paths and provenance hashes; empty when disabled.
    """
    if cfg.steering.clear_run is None:
        return {}, {}
    root = Path(cfg.steering.clear_run)
    selected = json.loads((root / "selection.json").read_text())
    if selected["status"] != "complete" or selected["mode"] != "projected" or selected["threshold"] != threshold:
        message = "Require completed projected training with the original frozen threshold"
        raise ValueError(message)
    if selected["checkpoint_sha256"]["baseline"] != baseline_hash:
        message = "CLEAR and original comparators have different baselines"
        raise ValueError(message)
    paths = {f"clear_projected_{name}": Path(selected[f"{name}_checkpoint"]) for name in ("best_a21", "guarded")}
    hashes = {str(root / name): digest(root / name) for name in ("selection.json", "plan.json", "manifest.json")}
    for name, path in paths.items():
        hashes[str(path)] = digest(path)
        if hashes[str(path)] != selected["checkpoint_sha256"][name.removeprefix("clear_projected_")]:
            message = "CLEAR checkpoint changed after selection"
            raise ValueError(message)
    return paths, hashes


def add_clear_jobs(
    cfg: DictConfig,
    paths: dict[str, Path],
    jobs: dict[str, Scorer],
    aliases: dict[str, str],
) -> None:
    """Load each distinct selected CLEAR checkpoint once, aliasing baseline fallbacks.

    Args:
        cfg: Model construction and runtime settings.
        paths: Already verified selected checkpoints.
        jobs: Existing unique scorer mapping, modified in place.
        aliases: Exact score aliases, modified in place.
    """
    seen = {digest(Path(cfg.steering.checkpoint) / "model.safetensors"): "baseline"}
    for name, path in paths.items():
        checksum = digest(path)
        if checksum in seen:
            aliases[name] = seen[checksum]
        else:
            model = build_model(cfg).requires_grad_(False).eval()
            model.load_state_dict(load_file(path), strict=True)
            jobs[name] = Scorer(model, jobs["baseline"].layer)
            seen[checksum] = name


def save_report(output: Path, rows: list[ExternalRecording], scores: dict[str, np.ndarray], threshold: float) -> None:
    """Persist complete pooled and per-attack metrics using unchanged metric functions.

    Args:
        output: Exclusively owned evaluation directory.
        rows: Complete official development inventory.
        scores: Aligned finite probabilities for all methods.
        threshold: Original frozen threshold.
    """
    result = metrics(rows, scores, threshold)
    write_json(
        output / "report.json",
        {
            "status": "complete",
            "dataset": "ASVspoof 5 full development",
            "n": len(rows),
            "threshold": threshold,
            "results": result,
            "note": (
                "Full development includes the 576-recording selection guard. No reselection or recalibration. "
                "Each attack shares all development genuine recordings."
            ),
        },
    )
    with (output / "comparison.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "method",
                "group",
                "n_bonafide",
                "n_spoof",
                "eer_percent",
                "min_dcf",
                "fixed_dcf",
                "bonafide_rejection_percent",
                "spoof_acceptance_percent",
            ],
        )
        for method, groups in result.items():
            for group, row in groups.items():
                writer.writerow(
                    [
                        method,
                        group,
                        row["n_bonafide"],
                        row["n_spoof"],
                        100 * row["eer"],
                        row["min_dcf"],
                        row["fixed_dcf"],
                        100 * row["bonafide_rejection"],
                        100 * row["spoof_acceptance"],
                    ],
                )


def run(cfg: DictConfig, output: Path) -> None:
    """Score all development recordings under frozen A21, OY and optional CLEAR models.

    Args:
        cfg: Shared evaluation configuration using evaluation=asvspoof5_dev.
        output: Locked output directory owned by the common evaluation lifecycle.
    """
    settings = cfg.steering
    rows = development_rows(cfg)
    manifest = inventory(rows)
    identity = provenance(cfg, manifest)
    threshold = float(json.loads((Path(settings.reference_run) / "selection.json").read_text())["threshold"])
    baseline_hash = identity["sources_sha256"][str(Path(settings.checkpoint) / "model.safetensors")]
    paths, hashes = clear_artifacts(cfg, baseline_hash, threshold)
    identity["clear_artifacts"] = hashes
    identity["dev_sources"] = {
        str(path): digest(path)
        for path in [
            *Path("src/dev_eval").glob("*.py"),
            Path("configs/steering/asvspoof5_dev.yaml"),
            Path(cfg.data.root) / "ASVspoof5.dev.metadata.txt",
            Path("src/voxenes_eval/run.py"),
        ]
    }
    oy = fitted_artifacts(cfg) if settings.include_oy else None
    if oy is not None:
        root, fitted, selection, _ = oy
        if (
            fitted["sources_sha256"][str(Path(settings.checkpoint) / "model.safetensors")] != baseline_hash
            or selection["threshold"] != threshold
        ):
            message = "OY baseline or threshold differs from the A21 comparison"
            raise ValueError(message)
        identity["oy_artifacts"] = {
            str(p): digest(p)
            for p in [
                root / "selection.json",
                root / "provenance.json",
                root / "plan.json",
                *root.glob("directions_layer_*.safetensors"),
            ]
        }
        identity["methods"] += list(OY_METHODS[1:])
    identity["methods"] += list(paths)
    identity["adaptation"] = "Frozen selected weights; full development reporting only, no refitting or recalibration."
    validate_identity(output, identity)
    write_json(output / "manifest.json", manifest)
    write_json(output / "provenance.json", {"identity": identity, "config": OmegaConf.to_container(cfg, resolve=True)})
    if settings.prepare_only:
        log.info("Prepared full development: %s recordings", len(rows))
        return
    jobs, aliases = build_scorers(cfg)
    smoke = smoke_check(jobs, aliases, cfg)
    if oy is not None:
        root, _, selection, _ = oy
        extra, extra_aliases = inference_jobs(cfg, root, selection, jobs["baseline"].model)
        jobs.update({key: value for key, value in extra.items() if key != "baseline"})
        aliases.update(extra_aliases)
    add_clear_jobs(cfg, paths, jobs, aliases)
    samples = smoke_samples(rows)
    waves = torch.stack([wave for _, wave in ExternalAudio(samples, int(cfg.data.num_samples))])
    trial_scores = score_batch(jobs, aliases, waves, int(settings.batch_size))
    if any(not np.isfinite(value).all() for value in trial_scores.values()):
        message = "Nonfinite development smoke probabilities"
        raise ValueError(message)
    smoke.update({"status": "passed", "dev_recordings": len(samples), "aliases": aliases})
    write_json(output / "smoke.json", smoke)
    if settings.smoke_only:
        return
    scoring_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    assert isinstance(scoring_cfg, DictConfig)
    scoring_cfg.steering.methods = identity["methods"]
    scores = stream(rows, jobs, aliases, scoring_cfg, output)
    save_report(output, rows, scores, threshold)
    write_json(output / "progress.json", {"status": "complete", "processed": len(rows), "total": len(rows)})
    log.info("Completed full ASVspoof development: %s", output)
