"""Resumable, generation-only full ASVspoof 5 reasoning evaluation."""

import argparse
import fcntl
import json
import logging
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from src.alm_reasoning.artifacts import digest, restore_chunks, write_json
from src.alm_reasoning.metrics import Cohorts, parse_answer
from src.alm_reasoning.runtime import ReasoningALM
from src.steering.protocol import Trial, read_trials

LOGGER = logging.getLogger(__name__)


def smoke(model: ReasoningALM, cfg: dict[str, Any], output: Path) -> None:
    """Verify cached decoding on support recordings before any full evaluation.

    Args:
        model: Strictly restored frozen model.
        cfg: Run configuration.
        output: Artifact directory.
    """
    manifest = json.loads(Path(cfg["support_manifest"]).read_text())
    selected = [Trial(**next(row for row in manifest["support"] if row["label"] == label)) for label in (0, 1)]
    audio = model.encode([Path(t.audio_dir) / f"{t.utterance}.flac" for t in selected])
    cached = model.generate(audio, 8)
    uncached = model.generate(audio, 8, use_cache=False)
    if [row["token_ids"] for row in cached] != [row["token_ids"] for row in uncached]:
        message = "Cached and uncached greedy decoding differ in the smoke check"
        raise ValueError(message)
    started = time.perf_counter()
    generated = model.generate(audio, cfg["max_new_tokens"])
    write_json(
        output / "smoke.json",
        {
            "status": "passed",
            "load": model.audit,
            "cache_token_parity": True,
            "audio_shape": list(audio.shape),
            "generation_seconds": time.perf_counter() - started,
            "support_predictions": [
                {"utterance": trial.utterance, "label": trial.label, "prediction": parse_answer(row["text"]), **row}
                for trial, row in zip(selected, generated, strict=True)
            ],
            "note": "Support-only engineering smoke; not an estimate of test accuracy.",
        },
    )
    LOGGER.info("Support smoke passed: strict load, frozen parameters, cache token parity")


def evaluate(cfg: dict[str, Any], *, smoke_only: bool = False) -> None:
    """Run immutable full-cohort inference, resuming only identical experiments.

    Args:
        cfg: Explicit frozen experiment settings.
        smoke_only: Exit after engineering checks without test inference.
    """
    output = Path(cfg["output"])
    output.mkdir(parents=True, exist_ok=True)
    chunks = output / "chunks"
    chunks.mkdir(exist_ok=True)
    with (output / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(4)
        torch.manual_seed(cfg["seed"])
        root = Path(cfg["dataset_root"])
        inputs = [
            Path(cfg["checkpoint"]),
            Path(cfg["model_config"]),
            Path(cfg["support_manifest"]),
            root / "last_eval/ASVspoof5.eval.track_1.tsv",
            Path(cfg["dataset_source"]),
            Path("src/alm_steering/runtime.py"),
            Path("src/alm_steering/run.py"),
            Path("src/steering/protocol.py"),
            *sorted(Path("src/alm_reasoning").glob("*.py")),
            *sorted(Path(cfg["source_package"]).rglob("*.py")),
        ]
        LOGGER.info("Hashing checkpoint, source and protocol for resumable provenance")
        identity = {"config": cfg, "sha256": {str(path): digest(path) for path in inputs}}
        provenance = output / "provenance.json"
        if provenance.exists() and json.loads(provenance.read_text()) != identity:
            message = "Run identity changed; use a new output directory"
            raise ValueError(message)
        write_json(provenance, identity)
        trials = read_trials(root, "eval")
        cohorts = Cohorts()
        offset, seconds = restore_chunks(chunks, trials, cohorts)
        write_json(output / "progress.json", {"status": "loading", "completed": offset, "total": len(trials)})
        if offset == len(trials):
            write_json(
                output / "report.json",
                {
                    "status": "complete",
                    "completed": offset,
                    "total": len(trials),
                    "inference_seconds": seconds,
                    "metrics": cohorts.report(),
                },
            )
            return
        started = time.perf_counter()
        model = ReasoningALM(cfg)
        load_seconds = time.perf_counter() - started
        write_json(output / "load.json", {**model.audit, "load_seconds": load_seconds})
        smoke(model, cfg, output)
        if smoke_only:
            write_json(
                output / "progress.json",
                {"status": "smoke_complete", "completed": offset, "total": len(trials)},
            )
            return
        for start in range(offset, len(trials), cfg["chunk_size"]):
            selected = trials[start : start + cfg["chunk_size"]]
            predictions = []
            started = time.perf_counter()
            torch.cuda.reset_peak_memory_stats(model.device)
            for index in range(0, len(selected), cfg["batch_size"]):
                batch = selected[index : index + cfg["batch_size"]]
                audio = model.encode([Path(t.audio_dir) / f"{t.utterance}.flac" for t in batch])
                rows = model.generate(audio, cfg["max_new_tokens"])
                for trial, row in zip(batch, rows, strict=True):
                    prediction = parse_answer(row["text"])
                    predictions.append({"utterance": trial.utterance, "prediction": prediction, **row})
                    cohorts.add(trial, prediction)
            torch.cuda.synchronize(model.device)
            elapsed = time.perf_counter() - started
            seconds += elapsed
            completed = start + len(selected)
            write_json(
                chunks / f"{start:09d}.json",
                {
                    "start": start,
                    "wall_seconds": elapsed,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(model.device),
                    "predictions": predictions,
                },
            )
            report = {
                "status": "complete" if completed == len(trials) else "incomplete",
                "completed": completed,
                "total": len(trials),
                "inference_seconds": seconds,
                "seconds_per_recording": seconds / completed,
                "estimated_remaining_hours": (len(trials) - completed) * seconds / completed / 3600,
                "note": "Partial metrics cover processed records only, not the full benchmark.",
                "metrics": cohorts.report(),
            }
            write_json(output / "progress.json", report)
            LOGGER.info(
                "%s/%s records; %.1fs/chunk; ETA %.1fh",
                completed,
                len(trials),
                elapsed,
                report["estimated_remaining_hours"],
            )
        report["note"] = "Complete official evaluation; attack groups include all bonafide in the respective cohort."
        write_json(output / "report.json", report)


def main() -> None:
    """Read a fixed YAML configuration and start or resume evaluation."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/alm_reasoning/default.yaml"))
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    evaluate(yaml.safe_load(args.config.read_text()), smoke_only=args.smoke_only)


if __name__ == "__main__":
    main()
