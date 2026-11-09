"""Evaluate immutable speaker adaptations with cached original comparators."""

import csv
import hashlib
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from src.speaker_steering.adaptation import FAMILIES, transform
from src.speaker_steering.protocol import records
from src.steering.evaluation import attack_report
from src.steering.external_eval import digest, external_results, stream, validate_identity
from src.steering.external_protocol import ExternalAudio, ExternalRecording, inventory
from src.steering.finetune import build_model
from src.steering.full_eval import Scorer, load_chunks, score_batch
from src.steering.protocol import read_trials, speaker_partition
from src.steering.run import write_json

log = logging.getLogger(__name__)
METHODS = (
    "baseline",
    *FAMILIES,
    *(f"{name}_nonzero" for name in FAMILIES),
    "speaker_sae_random",
    "speaker_sae_reversed",
)


def speaker_cohorts(rows: list[ExternalRecording], plan: dict[str, Any]) -> dict[str, np.ndarray]:
    """Construct evaluation masks excluding every sibling of adaptation audio.

    Args:
        rows: Full prepared-segment inventory.
        plan: Frozen source ownership and selected speaker.

    Returns:
        Inclusive diagnostic, source-held-out, other-speaker and target subsets.
    """
    blocked = set(plan["excluded_sources"])
    heldout = np.array([row.source not in blocked for row in rows])
    other = np.array([row.speaker != plan["speaker"] for row in rows])
    target = np.array(
        [row.label == 0 or (row.speaker == plan["speaker"] and row.group == "elevenlabs") for row in rows],
    )
    indices = np.arange(len(rows))
    return {
        "inclusive_diagnostic": indices,
        "source_heldout": indices[heldout],
        "other_speakers": indices[other & heldout],
        "target_voice_and_heldout_bonafide": indices[target & heldout],
    }


def cached_scores(root: Path, count: int) -> dict[str, np.ndarray]:
    """Load complete comparator scores, refusing an unfinished run.

    Args:
        root: Original run with provenance and contiguous score chunks.
        count: Expected inventory size.

    Returns:
        Full aligned score vectors for the original nine methods.
    """
    identity = json.loads((root / "provenance.json").read_text())["identity"]
    scores = {name: np.full(count, np.nan, dtype=np.float32) for name in identity["methods"]}
    if load_chunks(root / "chunks", scores) != count:
        message = "Comparator is incomplete; resume this evaluation after it finishes (new scores are retained)"
        raise ValueError(message)
    return scores


def results(
    rows: list[ExternalRecording],
    scores: dict[str, np.ndarray],
    threshold: float,
    dataset: str,
    plan: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Apply existing metrics to prespecified complete comparison cohorts.

    Args:
        rows: Full dataset in score order.
        scores: Original comparators and new immutable speaker interventions.
        threshold: Original development threshold, never recalibrated.
        dataset: ElevenLabs, WildSpoof or full ASVspoof 5.
        plan: Frozen source exclusions for ElevenLabs.
        seed: Original ASVspoof speaker-partition seed.

    Returns:
        Complete cohort metrics, retaining original attack/generator breakdowns.
    """
    if dataset == "wildspoof_2026":
        return external_results(rows, scores, threshold, dataset)
    if dataset == "elevenlabs":
        output = {}
        for name, indices in speaker_cohorts(rows, plan).items():
            subset = [rows[i] for i in indices]
            values = {method: value[indices] for method, value in scores.items()}
            for cohort, result in external_results(subset, values, threshold, dataset).items():
                output[f"{name}/{cohort}"] = result
        return output
    cohorts = {
        "full_eval": np.arange(len(rows)),
        "original_heldout_speakers": np.flatnonzero([speaker_partition(row.speaker, seed) == "test" for row in rows]),
    }
    output = {}
    for name, indices in cohorts.items():
        trials = [rows[i].trial() for i in indices]
        metrics = {method: attack_report(trials, value[indices], threshold) for method, value in scores.items()}
        baseline = metrics["baseline"]["pooled"]["eer"]
        for value in metrics.values():
            value["pooled"]["delta_eer_pp"] = 100 * (value["pooled"]["eer"] - baseline)
        output[name] = {"status": "complete", "n": len(indices), "metrics": metrics}
    return output


def fitted_artifacts(cfg: DictConfig) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate immutable fitting inputs before any evaluation audio is loaded.

    Args:
        cfg: Fitting artifacts and fixed preprocessing settings.

    Returns:
        Artifact directory, identity, selection and source plan.
    """
    settings = cfg.steering
    root = Path(settings.adaptation_run)
    fitted = json.loads((root / "provenance.json").read_text())["identity"]
    for mapping in ("sources_sha256", "speaker_sources"):
        for name, expected in fitted[mapping].items():
            if digest(Path(name)) != expected:
                message = f"Fitted source changed: {name}"
                raise ValueError(message)
    if digest(Path("configs/steering/elevenlabs_speaker.yaml")) != fitted["speaker_config_sha256"]:
        message = "Speaker configuration changed since fitting"
        raise ValueError(message)
    selection = json.loads((root / "selection.json").read_text())
    plan = json.loads((root / "plan.json").read_text())
    if plan != fitted["speaker_plan"] or selection["test_scores_used"]:
        message = "Invalid fit plan or test-dependent selection"
        raise ValueError(message)
    if int(cfg.seed) != fitted["seed"] or int(cfg.data.num_samples) != fitted["num_samples"]:
        message = "Inference seed or preprocessing differs from fitting"
        raise ValueError(message)
    return root, fitted, selection, plan


def evaluation_inventory(
    cfg: DictConfig,
    dataset: str,
    reference: Path,
    prior: dict[str, Any],
) -> tuple[list[ExternalRecording], dict[str, Any]]:
    """Check ordered evaluation metadata against original comparator artifacts.

    Args:
        cfg: Dataset root.
        dataset: Requested dataset identifier.
        reference: Original comparator output directory.
        prior: Original comparator provenance.

    Returns:
        Ordered audio records and auditable inventory.
    """
    if dataset == "asvspoof5":
        protocol = Path(cfg.data.root) / "last_eval/ASVspoof5.eval.track_1.tsv"
        previous_protocol = [
            value
            for name, value in prior["identity"]["sources_sha256"].items()
            if name.endswith("ASVspoof5.eval.track_1.tsv")
        ]
        if previous_protocol != [digest(protocol)]:
            message = "ASVspoof protocol differs from cached comparator order"
            raise ValueError(message)
        rows = [
            ExternalRecording(
                row.utterance,
                str(Path(row.audio_dir) / f"{row.utterance}.flac"),
                row.label,
                row.attack,
                speaker=row.speaker,
                source=row.source,
            )
            for row in read_trials(Path(cfg.data.root), "eval")
        ]
        if len(rows) != prior["n_recordings"]:
            message = "Full ASVspoof recording count differs"
            raise ValueError(message)
        manifest = {
            "recordings": [asdict(row) for row in rows],
            "protocol_sha256": digest(protocol),
            "audio_identity_policy": "Exact official protocol; original run did not hash individual audio",
        }
    else:
        old_manifest = json.loads((reference / "manifest.json").read_text())
        rows = records(old_manifest["recordings"])
        manifest = inventory(rows)
        if manifest != old_manifest:
            message = "External audio inventory changed since comparator preparation"
            raise ValueError(message)
    return rows, manifest


def write_comparison(output: Path, report: dict[str, Any]) -> None:
    """Export complete cohort and method comparisons in a flat table.

    Args:
        output: Evaluation directory.
        report: Completed cohort metrics.
    """
    with (output / "comparison.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cohort", "group", "method", "eer_percent", "min_dcf", "bonafide_rejection_percent"])
        for cohort, result in report.items():
            for method, groups in result.get("metrics", {}).items():
                for group, metric in groups.items():
                    writer.writerow(
                        [
                            cohort,
                            group,
                            method,
                            100 * metric["eer"],
                            metric["min_dcf"],
                            100 * metric["bonafide_rejection"],
                        ],
                    )


def inference_jobs(
    cfg: DictConfig,
    root: Path,
    selection: dict[str, Any],
    model: Any,
) -> tuple[dict[str, Scorer], dict[str, str]]:
    """Restore selected directions and prespecified controls without test labels.

    Args:
        cfg: Device and dictionary settings.
        root: Fitted direction directory.
        selection: Frozen layer and strength choices.
        model: Shared frozen detector.

    Returns:
        Unique inference jobs and exact-zero baseline aliases.
    """
    settings = cfg.steering
    jobs = {"baseline": Scorer(model, int(settings.layers[0]))}
    aliases = {}
    for method in METHODS[1:]:
        control = method in {"speaker_sae_random", "speaker_sae_reversed"}
        family = "speaker_sae" if control else method.removesuffix("_nonzero")
        key = "nonzero_diagnostics" if control or method.endswith("_nonzero") else "selected"
        row = selection[key][family]
        if row["strength"] == 0:
            aliases[method] = "baseline"
        else:
            operation = method if control else family
            jobs[method] = Scorer(model, row["layer"], transform(cfg, root, operation, row["layer"], row["strength"]))
    return jobs, aliases


def evaluate(cfg: DictConfig, output: Path) -> None:
    """Score all recordings after checking frozen fit and comparator identities.

    Args:
        cfg: Selected adaptation, evaluation dataset and inference settings.
        output: Exclusively locked resumable output directory.
    """
    settings = cfg.steering
    root, fitted, selection, plan = fitted_artifacts(cfg)
    dataset = str(settings.evaluation_dataset)
    if dataset not in {"elevenlabs", "wildspoof_2026", "asvspoof5"}:
        message = f"Unknown dataset: {dataset}"
        raise ValueError(message)
    baseline_key = {"elevenlabs": "elevenlabs", "wildspoof_2026": "wildspoof", "asvspoof5": "asvspoof"}[dataset]
    reference = Path(settings[f"baseline_{baseline_key}_run"])
    prior = json.loads((reference / "provenance.json").read_text())
    rows, manifest = evaluation_inventory(cfg, dataset, reference, prior)
    for key in ("num_samples", "batch_size", "seed"):
        if prior["identity"][key] != fitted[key]:
            message = f"Comparator setting differs: {key}"
            raise ValueError(message)
    checkpoint = str(Path(settings.checkpoint) / "model.safetensors")
    if prior["identity"]["sources_sha256"][checkpoint] != fitted["sources_sha256"][checkpoint]:
        message = "Comparator checkpoint differs"
        raise ValueError(message)
    artifacts = [root / name for name in ("selection.json", "plan.json", "provenance.json")]
    artifacts.extend(root.glob("directions_layer_*.safetensors"))
    identity = {
        "adaptation_artifacts": {str(path): digest(path) for path in artifacts},
        "manifest_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
        "comparator_provenance_sha256": digest(reference / "provenance.json"),
        "dataset": dataset,
        "methods": list(METHODS),
        "batch_size": int(settings.batch_size),
        "num_samples": int(cfg.data.num_samples),
        "seed": int(cfg.seed),
        "torch": str(torch.__version__),
    }
    validate_identity(output, identity)
    write_json(output / "manifest.json", manifest)
    write_json(output / "provenance.json", {"identity": identity, "config": OmegaConf.to_container(cfg, resolve=True)})
    if settings.prepare_only:
        return
    model = build_model(cfg).requires_grad_(False).eval()
    jobs, aliases = inference_jobs(cfg, root, selection, model)
    scoring_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    assert isinstance(scoring_cfg, DictConfig)
    scoring_cfg.steering.methods = list(METHODS)
    waves = torch.stack([wave for _, wave in ExternalAudio(rows[:8], int(cfg.data.num_samples))])
    smoke = score_batch(jobs, aliases, waves, int(settings.batch_size))
    if any(not np.isfinite(value).all() for value in smoke.values()):
        message = "Nonfinite speaker evaluation smoke scores"
        raise ValueError(message)
    write_json(output / "smoke.json", {"status": "passed", "n": len(waves), "aliases": aliases})
    if settings.smoke_only:
        return
    scores = stream(rows, jobs, aliases, scoring_cfg, output)
    original = cached_scores(reference, len(rows))
    max_difference = float(np.max(np.abs(scores["baseline"] - original["baseline"])))
    np.testing.assert_allclose(scores["baseline"], original["baseline"], atol=float(settings.smoke_score_atol), rtol=0)
    for name, target in aliases.items():
        np.testing.assert_array_equal(scores[name], scores[target])
    combined = {**original, **scores}
    report = results(rows, combined, float(selection["threshold"]), dataset, plan, int(cfg.seed))
    write_json(
        output / "report.json",
        {
            "status": "complete",
            "dataset": dataset,
            "speaker": plan["speaker"],
            "threshold": selection["threshold"],
            "results": report,
            "baseline_cache_max_absolute_difference": max_difference,
            "note": "Selection frozen before evaluation. Inclusive ElevenLabs overlaps adaptation; use source_heldout.",
        },
    )
    write_comparison(output, report)
    write_json(output / "progress.json", {"status": "complete", "processed": len(rows), "total": len(rows)})
    log.info("Completed %s for speaker %s: %s", dataset, plan["speaker"], output)
