"""Split a resumable full evaluation into disjoint, batch-aligned GPU shards."""

import argparse
import fcntl
import hashlib
import json
import logging
import signal
import subprocess
import sys
import time
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from transformers import set_seed

from src.steering.full_eval import (
    EvalAudio,
    build_scorers,
    final_report,
    load_chunks,
    save_chunk,
    score_batch,
    smoke_check,
)
from src.steering.protocol import read_trials

log = logging.getLogger(__name__)


def atomic_json(path: Path, content: Any) -> None:
    """Publish JSON atomically so the coordinator never reads partial progress.

    Args:
        path: Run-owned destination with exactly one writer.
        content: Strictly JSON-serializable report.
    """
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def split_ranges(start: int, stop: int, workers: int, batch_size: int) -> list[tuple[int, int]]:
    """Partition remaining records without changing original batch boundaries.

    Args:
        start: End of the already completed contiguous prefix.
        stop: Total number of protocol recordings.
        workers: Number of nonempty shards.
        batch_size: Original inference batch size.

    Returns:
        Adjacent half-open ranges. Only the last shard may end in a partial batch.
    """
    if batch_size < 1 or workers < 1 or not 0 <= start < stop or start % batch_size:
        message = "Require positive budgets and a batch-aligned, nonempty remaining range"
        raise ValueError(message)
    batches = (stop - start) // batch_size
    if batches < workers:
        message = "Too few full batches for the requested worker count"
        raise ValueError(message)
    boundaries = [
        start + round((stop - start) * index / (workers * batch_size)) * batch_size for index in range(workers)
    ]
    boundaries.append(stop)
    return list(pairwise(boundaries))


def restore_shard(directory: Path, start: int, stop: int, methods: list[str]) -> tuple[dict[str, np.ndarray], int]:
    """Restore a contiguous shard prefix, rejecting overlaps and corrupt chunks.

    Args:
        directory: Shared chunk directory; each worker owns a distinct range.
        start: First absolute protocol index owned by this shard.
        stop: Exclusive final protocol index owned by this shard.
        methods: Required method names.

    Returns:
        Shard-local score arrays and absolute next index to process.
    """
    scores = {name: np.full(stop - start, np.nan, dtype=np.float32) for name in methods}
    expected = start
    for path in sorted(directory.glob("*.npz")):
        if path.name.endswith(".partial.npz"):
            continue
        # Filename bounds let workers ignore other workers' concurrently published files.
        lower, upper = map(int, path.stem.split("_"))
        if upper <= start or lower >= stop:
            continue
        with np.load(path, allow_pickle=False) as chunk:
            if (
                int(chunk["start"]) != lower
                or int(chunk["end"]) != upper
                or lower != expected
                or not lower < upper <= stop
                or set(chunk.files) != {"start", "end", *methods}
            ):
                message = f"Overlapping, noncontiguous, or invalid shard chunk: {path}"
                raise ValueError(message)
            for method in methods:
                values = chunk[method]
                if values.shape != (upper - lower,) or not np.isfinite(values).all():
                    message = f"Invalid scores in {path}: {method}"
                    raise ValueError(message)
                scores[method][lower - start : upper - start] = values
        expected = upper
    return scores, expected


def original_config(directory: Path, *, verify: bool) -> tuple[DictConfig, dict[str, Any]]:
    """Restore the original run configuration and optionally verify every source.

    Args:
        directory: Original evaluation directory, whose artifacts are preserved.
        verify: Hash all recorded code, model, selection, and protocol inputs.

    Returns:
        Original resolved configuration and provenance document.
    """
    provenance = json.loads((directory / "provenance.json").read_text())
    if verify:
        for filename, expected in provenance["identity"]["sources_sha256"].items():
            with Path(filename).open("rb") as handle:
                actual = hashlib.file_digest(handle, "sha256").hexdigest()
            if actual != expected:
                message = f"Original scoring input has changed: {filename}"
                raise ValueError(message)
    cfg = OmegaConf.create(provenance["config"])
    assert isinstance(cfg, DictConfig)
    return cfg, provenance


def worker(directory: Path, index: int) -> None:
    """Score only the assigned recording interval with unchanged frozen methods.

    Args:
        directory: Shared original run directory.
        index: Index into the persisted parallel assignment plan.
    """
    plan = json.loads((directory / "parallel_plan.json").read_text())
    assignment = plan["shards"][index]
    start, stop = int(assignment["start"]), int(assignment["stop"])
    cfg, _ = original_config(directory, verify=False)
    cfg.steering.device = assignment["device"]
    set_seed(int(cfg.seed))
    torch.set_num_threads(4)
    status_path = directory / "parallel" / f"worker_{index}.json"
    scores, processed = restore_shard(directory / "chunks", start, stop, list(cfg.steering.methods))
    state = {
        "device": assignment["device"],
        "start": start,
        "stop": stop,
        "processed": processed,
        "checkpointed": processed,
    }
    atomic_json(status_path, {**state, "status": "initializing"})
    if processed == stop:
        atomic_json(status_path, {**state, "status": "complete"})
        return
    jobs, aliases = build_scorers(cfg)
    smoke = smoke_check(jobs, aliases, cfg)
    atomic_json(directory / "parallel" / f"worker_{index}_smoke.json", smoke)
    log.info("Worker %s restored-score smoke passed: %s", index, smoke)
    trials = read_trials(Path(cfg.data.root), "eval")
    loader = DataLoader(
        EvalAudio(trials[:stop], int(cfg.data.num_samples), processed),
        batch_size=int(cfg.steering.batch_size),
        num_workers=int(cfg.steering.audio_workers),
        shuffle=False,
        pin_memory=True,
        multiprocessing_context="spawn" if cfg.steering.audio_workers else None,
    )
    checkpointed = processed
    initial = processed
    started: float | None = None
    last_log = 0.0
    for indices, waves in loader:
        if started is None:
            started = time.perf_counter()
        np.testing.assert_array_equal(indices.numpy(), np.arange(processed, processed + len(waves)))
        batch = score_batch(jobs, aliases, waves, int(cfg.steering.batch_size))
        end = processed + len(waves)
        for method, values in batch.items():
            scores[method][processed - start : end - start] = values
        processed = end
        if processed - checkpointed >= int(cfg.steering.chunk_size) or processed == stop:
            checkpointed = save_chunk(
                directory / "chunks",
                checkpointed,
                {name: values[checkpointed - start : processed - start] for name, values in scores.items()},
            )
        elapsed = time.perf_counter() - started
        if elapsed - last_log >= 30 or processed == stop:
            state.update(processed=processed, checkpointed=checkpointed)
            atomic_json(
                status_path,
                {
                    **state,
                    "status": "complete" if processed == stop else "scoring",
                    "scoring_seconds": elapsed,
                    "recordings_per_second_all_methods": (processed - initial) / elapsed,
                },
            )
            log.info("Worker %s: %s/%s records in shard", index, processed - start, stop - start)
            last_log = elapsed


def make_plan(directory: Path, devices: list[str], cfg: DictConfig, provenance: dict[str, Any]) -> dict[str, Any]:
    """Preserve completed chunks and persist a reproducible disjoint assignment.

    Args:
        directory: Original run directory; original scorer must be stopped first.
        devices: Distinct CUDA device names for the new workers.
        cfg: Verified original configuration.
        provenance: Original provenance document.

    Returns:
        New or verified existing plan, including the preserved prefix length.
    """
    path = directory / "parallel_plan.json"
    identity = hashlib.sha256((directory / "provenance.json").read_bytes()).hexdigest()
    code = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if path.exists():
        plan = json.loads(path.read_text())
        if (
            plan["original_provenance_sha256"] != identity
            or plan["orchestrator_sha256"] != code
            or [row["device"] for row in plan["shards"]] != devices
        ):
            message = "Parallel resume provenance, orchestration code, or devices differ"
            raise ValueError(message)
        return plan
    total = int(provenance["n_recordings"])
    scores = {str(name): np.full(total, np.nan, dtype=np.float32) for name in cfg.steering.methods}
    prefix = load_chunks(directory / "chunks", scores)
    ranges = split_ranges(prefix, total, len(devices), int(cfg.steering.batch_size))
    plan = {
        "original_provenance_sha256": identity,
        "orchestrator_sha256": code,
        "preserved_prefix": prefix,
        "total": total,
        "batch_size": int(cfg.steering.batch_size),
        "shards": [
            {"device": device, "start": start, "stop": stop}
            for device, (start, stop) in zip(devices, ranges, strict=True)
        ],
    }
    if (directory / "progress.json").exists():
        atomic_json(
            directory / "parallel" / "original_progress.json",
            json.loads((directory / "progress.json").read_text()),
        )
    atomic_json(path, plan)
    return plan


def monitor(directory: Path, plan: dict[str, Any], processes: list[subprocess.Popen[bytes]]) -> None:
    """Publish aggregate progress until workers finish, and reject failed runs.

    Args:
        directory: Original run artifact directory.
        plan: Persisted assignment plan.
        processes: Only the worker processes started by this coordinator.
    """
    last_log = 0.0
    while True:
        completed = int(plan["preserved_prefix"])
        checkpointed = completed
        workers = []
        for index, assignment in enumerate(plan["shards"]):
            path = directory / "parallel" / f"worker_{index}.json"
            row = (
                json.loads(path.read_text())
                if path.exists()
                else {
                    **assignment,
                    "status": "initializing",
                    "processed": assignment["start"],
                    "checkpointed": assignment["start"],
                }
            )
            completed += int(row["processed"]) - int(assignment["start"])
            checkpointed += int(row["checkpointed"]) - int(assignment["start"])
            workers.append(row)
        progress = {
            "status": "parallel_scoring",
            "processed": completed,
            "checkpointed": checkpointed,
            "total": plan["total"],
            "workers": workers,
        }
        atomic_json(directory / "progress.json", progress)
        if time.monotonic() - last_log >= 30:
            log.info("Parallel progress: %s/%s, checkpointed %s", completed, plan["total"], checkpointed)
            last_log = time.monotonic()
        codes = [process.poll() for process in processes]
        if all(code is not None for code in codes):
            if any(code != 0 for code in codes):
                atomic_json(directory / "progress.json", {**progress, "status": "failed", "worker_exit_codes": codes})
                message = f"Parallel workers failed: {codes}; completed chunks are preserved"
                raise RuntimeError(message)
            return
        time.sleep(5)


def coordinate(directory: Path, devices: list[str]) -> None:
    """Launch independent scoring workers and merge only after complete coverage.

    Args:
        directory: Existing full-eval run; the original process must be stopped.
        devices: Unique devices explicitly authorized for this run.
    """
    if len(devices) < 2 or len(set(devices)) != len(devices):
        message = "Choose at least two distinct devices"
        raise ValueError(message)
    work = directory / "parallel"
    work.mkdir(exist_ok=True)
    with (work / "coordinator.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        cfg, provenance = original_config(directory, verify=True)
        plan = make_plan(directory, devices, cfg, provenance)
        log.info("Parallel assignments: %s", plan)
        processes = []
        for index in range(len(devices)):
            with (work / f"worker_{index}.log").open("ab") as handle:
                processes.append(
                    subprocess.Popen(  # noqa: S603 -- Fixed Python module and argument vector; no shell execution.
                        [
                            sys.executable,
                            "-m",
                            "src.steering.parallel_eval",
                            "worker",
                            str(directory),
                            "--index",
                            str(index),
                        ],
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                    ),
                )
        try:
            monitor(directory, plan, processes)
        except KeyboardInterrupt:
            for process in processes:
                if process.poll() is None:
                    process.send_signal(signal.SIGINT)
            for process in processes:
                process.wait()
            raise
        scores = {str(name): np.full(int(plan["total"]), np.nan, dtype=np.float32) for name in cfg.steering.methods}
        complete = load_chunks(directory / "chunks", scores)
        if complete != int(plan["total"]):
            message = "Scoring ended without complete contiguous protocol coverage"
            raise RuntimeError(message)
        atomic_json(
            directory / "progress.json",
            {"status": "computing_metrics", "processed": complete, "total": complete},
        )
        final_report(read_trials(Path(cfg.data.root), "eval"), scores, cfg, directory)
        log.info("Complete full-eval report written: %s", directory)


def main() -> None:
    """Parse coordinator/worker commands, preserving the original experiment config."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("coordinate", "worker"))
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--devices", nargs="+", default=[])
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    directory = args.run_directory.resolve()
    if args.mode == "coordinate":
        coordinate(directory, args.devices)
    else:
        worker(directory, args.index)


if __name__ == "__main__":
    main()
