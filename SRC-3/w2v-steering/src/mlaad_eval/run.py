"""Frozen wav2vec2-AASIST comparison on original English MLAAD audio."""

import csv
import fcntl
import json
import logging
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import set_seed

from src.mlaad_eval.protocol import build_plan
from src.speaker_steering.evaluation import METHODS as OY_METHODS
from src.speaker_steering.evaluation import fitted_artifacts, inference_jobs
from src.steering.external_eval import digest, provenance, stream, validate_identity
from src.steering.external_protocol import ExternalAudio, ExternalRecording, inventory
from src.steering.full_eval import build_scorers, score_batch, smoke_check
from src.voxenes_eval.archive import write_status
from src.voxenes_eval.run import metrics, smoke_samples

log = logging.getLogger(__name__)


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
            "dataset": "MLAAD local English + corresponding M-AILABS originals",
            "threshold": threshold,
            "results": {"english": {"status": "complete", "n": len(rows), "metrics": result}},
            "note": (
                "Pooled metrics use every selected recording once, not a generator-macro average. "
                "Each generator is paired with all English bonafide recordings. "
                "Only local fake/en metadata rows and their unique English M-AILABS originals are used. "
                "All adaptations retain their previously frozen settings; no MLAAD fitting or calibration."
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


def run(cfg: DictConfig, output: Path) -> None:
    """Verify extraction, restore frozen adaptations and score original English.

    Args:
        cfg: Hydra steering=mlaad configuration.
        output: Exclusively locked output directory.
    """
    settings = cfg.steering
    rows, plan = build_plan(
        Path(settings.external_root),
        Path(settings.real_root),
    )
    if any(
        plan[key] != int(settings[expected])
        for key, expected in (
            ("n_recordings", "expected_recordings"),
            ("n_spoof", "expected_spoof"),
            ("n_bonafide", "expected_bonafide"),
            ("n_generators", "expected_generators"),
        )
    ):
        message = f"Expected {settings.expected_recordings} original-English records, found {len(rows)}"
        raise ValueError(message)
    write_status(output / "plan.json", plan)
    manifest = inventory(rows)
    identity = provenance(cfg, manifest)
    identity["mlaad_plan"] = plan
    identity["mlaad_sources"] = {
        str(path): digest(path)
        for path in [
            *sorted(Path("src/mlaad_eval").glob("*.py")),
            Path("configs/steering/mlaad.yaml"),
            Path("src/voxenes_eval/run.py"),
            Path("src/voxenes_eval/archive.py"),
        ]
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
        identity["adaptation"] = "Frozen existing A21 and OY adaptations; no MLAAD fitting, selection or calibration."
    validate_identity(output, identity)
    write_status(output / "manifest.json", manifest)
    write_status(
        output / "provenance.json",
        {"identity": identity, "config": OmegaConf.to_container(cfg, resolve=True)},
    )
    if settings.prepare_only:
        log.info("Prepared MLAAD English: %s recordings, %s generators", len(rows), plan["n_generators"])
        return
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
        message = "Nonfinite MLAAD smoke probabilities"
        raise ValueError(message)
    smoke.update({"status": "passed", "mlaad_recordings": len(samples), "aliases": aliases})
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
    log.info("Completed original English MLAAD: %s", output)


@hydra.main(version_base="1.3", config_path="../../configs", config_name="main")
def main(cfg: DictConfig) -> None:
    """Execute a locked, deterministic MLAAD evaluation.

    Args:
        cfg: Configuration using steering=mlaad.
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
