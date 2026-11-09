"""Frozen wav2vec2-AASIST comparison on original English VoxENES audio."""

import csv
import fcntl
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import set_seed

from src.alm_reasoning.metrics import summarize
from src.speaker_steering.evaluation import METHODS as OY_METHODS
from src.speaker_steering.evaluation import fitted_artifacts, inference_jobs
from src.steering.evaluation import attack_report
from src.steering.external_eval import digest, provenance, stream, validate_identity
from src.steering.external_protocol import ExternalAudio, ExternalRecording, inventory
from src.steering.full_eval import build_scorers, score_batch, smoke_check
from src.voxenes_eval.archive import write_status
from src.voxenes_eval.protocol import build_plan

log = logging.getLogger(__name__)


def metrics(rows: list[ExternalRecording], scores: dict[str, np.ndarray], threshold: float) -> dict[str, Any]:
    """Reuse existing metrics for the pooled set and each generator plus all real.

    Args:
        rows: Complete original-English recording inventory.
        scores: Finite bonafide-positive probabilities for each method.
        threshold: Frozen ASVspoof development threshold.

    Returns:
        Method-by-group metrics, with shared genuine references for generators.
    """
    if "baseline" not in scores or any(
        value.shape != (len(rows),) or not np.isfinite(value).all() for value in scores.values()
    ):
        message = "Require complete aligned scores including baseline"
        raise ValueError(message)
    trials = [row.trial() for row in rows]
    output: dict[str, Any] = {}
    for method, values in scores.items():
        result: dict[str, Any] = attack_report(trials, values, threshold)
        for group, metric in result.items():
            counts = [[0, 0, 0], [0, 0, 0]]
            for row, value in zip(rows, values, strict=True):
                if group == "pooled" or row.label == 0 or row.group == group:
                    counts[row.label][int(value < threshold)] += 1
            metric["classification"] = summarize(counts)
        output[method] = result
    for groups in output.values():
        for group, metric in groups.items():
            metric["delta_eer_pp"] = 100 * (metric["eer"] - output["baseline"][group]["eer"])
    return output


def save_report(output: Path, rows: list[ExternalRecording], scores: dict[str, np.ndarray], threshold: float) -> None:
    """Write complete pooled and generator results, retaining raw aligned chunks.

    Args:
        output: Run directory.
        rows: Complete evaluation records.
        scores: All method predictions, without any dataset-specific fitting.
        threshold: Original fixed threshold.
    """
    result = metrics(rows, scores, threshold)
    write_status(
        output / "report.json",
        {
            "status": "complete",
            "dataset": "VoxENES 2026: original English",
            "threshold": threshold,
            "results": {"original_english": {"status": "complete", "n": len(rows), "metrics": result}},
            "note": (
                "Pooled metrics use every selected recording once, not a generator-macro average. "
                "Each generator is paired with all English bonafide recordings. "
                "TTS fixed/random modes are combined; augmented and Spanish audio are excluded. "
                "OY nonzero methods are frozen exploratory diagnostics, not guarded selections."
            ),
        },
    )
    with (output / "comparison.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "group",
                "method",
                "n_bonafide",
                "n_spoof",
                "eer_percent",
                "delta_eer_pp",
                "min_dcf",
                "fixed_dcf",
                "bonafide_rejection_percent",
                "spoof_acceptance_percent",
                "accuracy",
                "precision",
                "recall",
                "f1",
            ],
        )
        for method, groups in result.items():
            for group, metric in groups.items():
                writer.writerow(
                    [
                        group,
                        method,
                        int(metric["n_bonafide"]),
                        int(metric["n_spoof"]),
                        100 * metric["eer"],
                        metric["delta_eer_pp"],
                        metric["min_dcf"],
                        metric["fixed_dcf"],
                        100 * metric["bonafide_rejection"],
                        100 * metric["spoof_acceptance"],
                        *[metric["classification"][key] for key in ("accuracy", "precision", "recall", "f1")],
                    ],
                )


def smoke_samples(rows: list[ExternalRecording]) -> list[ExternalRecording]:
    """Take two deterministic recordings from every available group.

    Args:
        rows: Complete sorted inventory.

    Returns:
        Small group-covering smoke-test inventory.
    """
    samples = []
    counts: Counter[str] = Counter()
    for row in rows:
        if counts[row.group] < 2:
            samples.append(row)
            counts[row.group] += 1
    return samples


def run(cfg: DictConfig, output: Path) -> None:
    """Verify extraction, restore frozen adaptations and score original English.

    Args:
        cfg: Hydra steering=voxenes configuration.
        output: Exclusively locked output directory.
    """
    settings = cfg.steering
    rows, plan = build_plan(
        Path(settings.external_root),
        Path(settings.archive_path),
        verify_audio=not settings.prepare_only,
    )
    if len(rows) != int(settings.expected_recordings):
        message = f"Expected {settings.expected_recordings} original-English records, found {len(rows)}"
        raise ValueError(message)
    write_status(output / "plan.json", plan)
    if settings.prepare_only:
        log.info("Prepared original English: %s", {k: v for k, v in plan.items() if k != "recordings"})
        return
    manifest = inventory(rows)
    identity = provenance(cfg, manifest)
    identity["archive"] = plan["archive"]
    identity["voxenes_sources"] = {
        str(path): digest(path)
        for path in [*sorted(Path("src/voxenes_eval").glob("*.py")), Path("configs/steering/voxenes.yaml")]
    }
    oy = fitted_artifacts(cfg) if settings.include_oy else None
    if oy is not None:
        root, fitted, selection, _ = oy
        checkpoint = str(Path(settings.checkpoint) / "model.safetensors")
        if any(identity[key] != fitted[key] for key in ("seed", "num_samples", "batch_size")) or (
            identity["sources_sha256"][checkpoint] != fitted["sources_sha256"][checkpoint]
        ):
            message = "OY fit and original comparator inference identities differ"
            raise ValueError(message)
        paths = [
            root / "selection.json",
            root / "plan.json",
            root / "provenance.json",
            *root.glob("directions_layer_*.safetensors"),
        ]
        identity["oy_artifacts"] = {str(path): digest(path) for path in paths}
        identity["methods"] = [*settings.methods, *OY_METHODS[1:]]
        identity["adaptation"] = "Frozen existing A21 and OY adaptations; no VoxENES fitting, selection or calibration."
    validate_identity(output, identity)
    write_status(output / "manifest.json", manifest)
    write_status(
        output / "provenance.json",
        {"identity": identity, "config": OmegaConf.to_container(cfg, resolve=True)},
    )
    threshold = float(json.loads((Path(settings.reference_run) / "selection.json").read_text())["threshold"])
    jobs, aliases = build_scorers(cfg)
    smoke = smoke_check(jobs, aliases, cfg)
    if oy is not None:
        root, _, selection, _ = oy
        if selection["threshold"] != threshold:
            message = "OY and A21 thresholds differ"
            raise ValueError(message)
        extra, extra_aliases = inference_jobs(cfg, root, selection, jobs["baseline"].model)
        jobs.update({key: value for key, value in extra.items() if key != "baseline"})
        aliases.update(extra_aliases)
    samples = smoke_samples(rows)
    waves = torch.stack([wave for _, wave in ExternalAudio(samples, int(cfg.data.num_samples))])
    sample_scores = score_batch(jobs, aliases, waves, int(settings.batch_size))
    if any(not np.isfinite(value).all() for value in sample_scores.values()):
        message = "Nonfinite VoxENES smoke probabilities"
        raise ValueError(message)
    smoke.update({"status": "passed", "voxenes_recordings": len(samples), "aliases": aliases})
    write_status(output / "smoke.json", smoke)
    if settings.smoke_only:
        return
    scoring_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    assert isinstance(scoring_cfg, DictConfig)
    scoring_cfg.steering.methods = identity["methods"]
    scores = stream(rows, jobs, aliases, scoring_cfg, output)
    for method, target in aliases.items():
        np.testing.assert_array_equal(scores[method], scores[target])
    save_report(output, rows, scores, threshold)
    write_status(output / "progress.json", {"status": "complete", "processed": len(rows), "total": len(rows)})
    log.info("Completed original English VoxENES: %s", output)


@hydra.main(version_base="1.3", config_path="../../configs", config_name="main")
def main(cfg: DictConfig) -> None:
    """Execute a locked, deterministic VoxENES evaluation.

    Args:
        cfg: Configuration using steering=voxenes.
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
