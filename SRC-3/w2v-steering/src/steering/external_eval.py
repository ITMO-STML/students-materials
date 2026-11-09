"""Resumable external evaluation of existing A21 adaptations without refitting."""

import csv
import fcntl
import hashlib
import json
import logging
import time
from collections import Counter
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from transformers import set_seed

from src.alm_reasoning.metrics import summarize
from src.steering.evaluation import attack_report
from src.steering.external_protocol import ExternalAudio, ExternalRecording, inventory, read_elevenlabs, read_wildspoof
from src.steering.full_eval import Scorer, build_scorers, load_chunks, save_chunk, score_batch, smoke_check
from src.steering.run import write_json

log = logging.getLogger(__name__)


def digest(path: Path) -> str:
    """Hash an artifact without loading model tensors.

    Args:
        path: Source file.

    Returns:
        SHA-256 hexadecimal digest.
    """
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def validate_identity(output: Path, identity: dict[str, Any]) -> None:
    """Reject changed inputs before reusing scores.

    Args:
        output: Run directory.
        identity: Current data, model and scoring identities.

    Raises:
        ValueError: Existing identity differs or chunks lack provenance.
    """
    path = output / "provenance.json"
    if path.exists():
        if json.loads(path.read_text())["identity"] != identity:
            message = "Resume identity differs; use a new output directory"
            raise ValueError(message)
    elif (output / "chunks").exists() and any((output / "chunks").glob("*.npz")):
        message = "Cannot restore chunks without provenance"
        raise ValueError(message)


def provenance(cfg: DictConfig, manifest: dict[str, Any]) -> dict[str, Any]:
    """Record identities and verify saved adaptation artifacts.

    Args:
        cfg: Experiment configuration.
        manifest: Audited recording inventory.

    Returns:
        Immutable inference identity.
    """
    settings = cfg.steering
    directories = [Path(settings[key]) for key in ("reference_run", "finetune_run", "midsteer_run")]
    manifests = [json.loads((path / "manifest.json").read_text()) for path in directories]
    if any(value != manifests[0] for value in manifests[1:]):
        message = "Saved adaptations used different manifests"
        raise ValueError(message)
    checkpoint = Path(settings.checkpoint) / "model.safetensors"
    sources = [checkpoint]
    for directory in directories:
        sources.extend(directory.glob("*.safetensors"))
        sources.extend(directory / name for name in ("selection.json", "manifest.json", "provenance.json"))
    for directory in ("src/steering", "src/model", "src/metrics", "src/data"):
        sources.extend(Path(directory).rglob("*.py"))
    sources.extend([Path("src/alm_reasoning/metrics.py"), Path("configs/steering/external_eval.yaml")])
    if settings.external_dataset == "wildspoof_2026":
        sources.append(Path(settings.external_root) / "protocol_test_label.tsv")
    hashes = {str(path): digest(path) for path in sorted(set(sources))}
    original = json.loads((directories[0] / "provenance.json").read_text())
    if hashes[str(checkpoint)] != original["checkpoint_sha256"]:
        message = "Baseline checkpoint differs from the saved adaptation"
        raise ValueError(message)
    prior = json.loads((Path(settings.verified_full_eval) / "provenance.json").read_text())["identity"]
    for path in sources:
        if (path.suffix == ".safetensors" or path.name == "selection.json") and prior["sources_sha256"].get(
            str(path),
        ) != hashes[str(path)]:
            message = f"Adaptation artifact differs from verified evaluation: {path}"
            raise ValueError(message)
    return {
        "manifest_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
        "sources_sha256": hashes,
        "dataset": str(settings.external_dataset),
        "num_samples": int(cfg.data.num_samples),
        "batch_size": int(settings.batch_size),
        "methods": list(settings.methods),
        "seed": int(cfg.seed),
        "precision": "FP32; no autocast",
        "torch": str(torch.__version__),
        "adaptation": "Existing ASVspoof 5 A21 support only; no external fitting, selection or calibration.",
    }


def external_results(
    rows: list[ExternalRecording],
    scores: dict[str, np.ndarray],
    threshold: float,
    dataset: str,
) -> dict[str, Any]:
    """Reuse existing detection metrics for complete external cohorts.

    Args:
        rows: Complete query or segment inventory.
        scores: Finite bonafide-positive scores for every method.
        threshold: Frozen ASVspoof development threshold.
        dataset: Dataset identifier.

    Returns:
        Unique-recording, datasource and trial-weighted comparisons.
    """
    if any(values.shape != (len(rows),) or not np.isfinite(values).all() for values in scores.values()):
        message = "External metrics require complete aligned finite scores"
        raise ValueError(message)
    indices = np.arange(len(rows))
    cohorts = {"all_unique_queries" if dataset == "wildspoof_2026" else "all_prepared_segments": indices}
    if dataset == "wildspoof_2026":
        cohorts["protocol_trial_weighted"] = np.repeat(indices, [row.multiplicity for row in rows])
        cohorts["excluding_asvspoof5"] = indices[[row.group != "asvspoof5" for row in rows]]
        cohorts.update(
            {
                f"datasource_{group}": indices[[row.group == group for row in rows]]
                for group in sorted({r.group for r in rows})
            },
        )
    else:
        cohorts["elevenlabs_generator_and_all_bonafide"] = indices[
            [row.label == 0 or row.group == "elevenlabs" for row in rows]
        ]
    result: dict[str, Any] = {}
    for cohort, selected in cohorts.items():
        subset = [rows[i] for i in selected]
        if {row.label for row in subset} != {0, 1}:
            result[cohort] = {
                "status": "not_estimable",
                "reason": "Cohort does not contain both classes",
                "n": len(subset),
            }
            continue
        trials = [row.trial() for row in subset]
        metrics: dict[str, Any] = {}
        for method, values in scores.items():
            probabilities = values[selected]
            metric: dict[str, Any] = attack_report(trials, probabilities, threshold)
            counts = [[0, 0, 0], [0, 0, 0]]
            for row, value in zip(subset, probabilities, strict=True):
                counts[row.label][int(value < threshold)] += 1
            metric["pooled"]["classification"] = summarize(counts)
            metrics[method] = metric
        baseline = metrics["baseline"]["pooled"]
        for value in metrics.values():
            pooled = value["pooled"]
            pooled["delta_eer_pp"] = 100 * (pooled["eer"] - baseline["eer"])
            pooled["delta_accuracy_pp"] = 100 * (
                pooled["classification"]["accuracy"] - baseline["classification"]["accuracy"]
            )
        result[cohort] = {"status": "complete", "n": len(subset), "metrics": metrics}
    return result


def stream(
    rows: list[ExternalRecording],
    jobs: dict[str, Scorer],
    aliases: dict[str, str],
    cfg: DictConfig,
    output: Path,
) -> dict[str, np.ndarray]:
    """Score every query once and commit resumable aligned chunks.

    Args:
        rows: Complete ordered metadata.
        jobs: Frozen inference jobs.
        aliases: Exact fallback mappings.
        cfg: Batch, worker and preprocessing settings.
        output: Identity-checked run directory.

    Returns:
        All method scores after complete coverage.
    """
    settings = cfg.steering
    scores = {str(name): np.full(len(rows), np.nan, dtype=np.float32) for name in settings.methods}
    chunks = output / "chunks"
    chunks.mkdir(exist_ok=True)
    processed = load_chunks(chunks, scores)
    initial = checkpointed = processed
    loader = DataLoader(
        ExternalAudio(rows, int(cfg.data.num_samples), processed),
        batch_size=int(settings.batch_size),
        num_workers=int(settings.audio_workers),
        shuffle=False,
        pin_memory=str(settings.device).startswith("cuda"),
        multiprocessing_context="spawn" if settings.audio_workers else None,
    )
    started = time.perf_counter()
    for indices, waves in loader:
        np.testing.assert_array_equal(indices.numpy(), np.arange(processed, processed + len(waves)))
        batch = score_batch(jobs, aliases, waves, int(settings.batch_size))
        end = processed + len(waves)
        for name, values in batch.items():
            if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
                message = f"Invalid probabilities: {name}"
                raise ValueError(message)
            scores[name][processed:end] = values
        processed = end
        if processed - checkpointed >= int(settings.chunk_size) or processed == len(rows):
            checkpointed = save_chunk(
                chunks,
                checkpointed,
                {name: values[checkpointed:processed] for name, values in scores.items()},
            )
            seconds = time.perf_counter() - started
            rate = (processed - initial) / seconds
            progress = {
                "status": "scoring",
                "processed": processed,
                "total": len(rows),
                "session_seconds": seconds,
                "recordings_per_second_all_methods": rate,
                "estimated_remaining_seconds": (len(rows) - processed) / rate,
            }
            write_json(output / "progress.json", progress)
            log.info("External evaluation: %s", progress)
    return scores


def save_report(rows: list[ExternalRecording], scores: dict[str, np.ndarray], cfg: DictConfig, output: Path) -> None:
    """Write final metrics only after complete scoring.

    Args:
        rows: Complete dataset inventory.
        scores: Aligned probabilities.
        cfg: Frozen threshold and dataset configuration.
        output: Run artifact directory.
    """
    settings = cfg.steering
    threshold = float(json.loads((Path(settings.reference_run) / "selection.json").read_text())["threshold"])
    results = external_results(rows, scores, threshold, str(settings.external_dataset))
    write_json(
        output / "report.json",
        {
            "status": "complete",
            "dataset": "WildSpoof 2026" if settings.external_dataset == "wildspoof_2026" else "ElevenLabs set",
            "threshold": threshold,
            "results": results,
            "note": (
                "Frozen A21 adaptations; no external training or tuning. "
                "Binary query antispoofing, not enrollment-aware SASV. "
                "ElevenLabs set includes other generators; prepared segments are correlated."
            ),
        },
    )
    with (output / "comparison.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "cohort",
                "group",
                "method",
                "eer_percent",
                "min_dcf",
                "fixed_dcf",
                "bonafide_rejection_percent",
                "spoof_acceptance_percent",
                "n_bonafide",
                "n_spoof",
            ],
        )
        for cohort, result in results.items():
            for method, groups in result.get("metrics", {}).items():
                for group, m in groups.items():
                    writer.writerow(
                        [
                            cohort,
                            group,
                            method,
                            100 * m["eer"],
                            m["min_dcf"],
                            m["fixed_dcf"],
                            100 * m["bonafide_rejection"],
                            100 * m["spoof_acceptance"],
                            int(m["n_bonafide"]),
                            int(m["n_spoof"]),
                        ],
                    )


def run(cfg: DictConfig, output: Path) -> None:
    """Prepare, smoke-check and evaluate a complete external cohort.

    Args:
        cfg: Hydra configuration using steering=external_eval.
        output: Exclusively owned run directory.
    """
    settings = cfg.steering
    dataset = str(settings.external_dataset)
    if dataset not in {"wildspoof_2026", "elevenlabs"}:
        message = f"Unknown external dataset: {dataset}"
        raise ValueError(message)
    reader = read_wildspoof if dataset == "wildspoof_2026" else read_elevenlabs
    rows = reader(Path(settings.external_root))
    manifest = inventory(rows)
    identity = provenance(cfg, manifest)
    validate_identity(output, identity)
    write_json(output / "manifest.json", manifest)
    write_json(output / "provenance.json", {"identity": identity, "config": OmegaConf.to_container(cfg, resolve=True)})
    if settings.prepare_only:
        log.info("Prepared %s: %d records", dataset, len(rows))
        return
    jobs, aliases = build_scorers(cfg)
    smoke = smoke_check(jobs, aliases, cfg)
    selected = []
    counts: Counter[tuple[str, int]] = Counter()
    for row in rows:
        group = (row.group, row.label)
        if counts[group] < 2:
            selected.append(row)
            counts[group] += 1
    waves = torch.stack([wave for _, wave in ExternalAudio(selected, int(cfg.data.num_samples))])
    started = time.perf_counter()
    values = score_batch(jobs, aliases, waves, int(settings.batch_size))
    for name, probabilities in values.items():
        if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
            message = f"External smoke failed: {name}"
            raise ValueError(message)
    smoke.update(
        {
            "status": "passed",
            "external_records": len(selected),
            "external_seconds_all_methods": time.perf_counter() - started,
        },
    )
    write_json(output / "smoke.json", smoke)
    log.info("Smoke passed: %s", smoke)
    if settings.smoke_only:
        return
    scores = stream(rows, jobs, aliases, cfg, output)
    for name, target in aliases.items():
        np.testing.assert_array_equal(scores[name], scores[target])
    save_report(rows, scores, cfg, output)
    write_json(output / "progress.json", {"status": "complete", "processed": len(rows), "total": len(rows)})
    log.info("Completed %s: %s", dataset, output)


@hydra.main(version_base="1.3", config_path="../../configs", config_name="main")
def main(cfg: DictConfig) -> None:
    """Run a frozen comparison without changing saved selections.

    Args:
        cfg: Composed experiment configuration.
    """
    set_seed(int(cfg.seed))
    torch.set_num_threads(4)
    output = Path(cfg.steering.resume_dir or cfg.paths.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(cfg, output)


if __name__ == "__main__":
    main()
