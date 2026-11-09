"""Shared-CLI parallel orchestration with unchanged ASVspoof scoring helpers."""

from __future__ import annotations

import fcntl
import logging
import signal
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from omegaconf import DictConfig

log = logging.getLogger(__name__)


def worker_command(directory: Path, index: int) -> list[str]:
    """Construct a worker invocation of the same public evaluation entrypoint.

    Args:
        directory: Existing ASVspoof run directory.
        index: Worker index in its frozen parallel plan.

    Returns:
        Python argument vector, without shell interpretation.
    """
    return [
        sys.executable,
        "-m",
        "src.eval",
        "evaluation=asvspoof5_worker",
        f"evaluation.run_directory={directory}",
        f"evaluation.worker_index={index}",
        f"hydra.run.dir={directory / 'parallel' / f'hydra_worker_{index}'}",
    ]


def coordinate(cfg: DictConfig, output: Path) -> None:
    """Restore a recorded run and launch workers through eval.py presets.

    Args:
        cfg: Parallel preset with explicitly selected devices.
        output: Original run directory, already locked by the shared API.
    """
    from src.steering.full_eval import final_report, load_chunks
    from src.steering.parallel_eval import atomic_json, make_plan, monitor, original_config
    from src.steering.protocol import read_trials

    devices = list(map(str, cfg.evaluation.devices))
    work = output / "parallel"
    work.mkdir(exist_ok=True)
    with (work / "coordinator.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        original, provenance = original_config(output, verify=True)
        total = int(provenance["n_recordings"])
        scores = {str(name): np.full(total, np.nan, dtype=np.float32) for name in original.steering.methods}
        if (output / "parallel_plan.json").exists() or load_chunks(output / "chunks", scores) != total:
            plan = make_plan(output, devices, original, provenance)
            processes = []
            try:
                for index in range(len(devices)):
                    with (work / f"worker_{index}.log").open("ab") as handle:
                        processes.append(
                            subprocess.Popen(  # noqa: S603 - fixed local Python argument vector
                                worker_command(output, index),
                                stdout=handle,
                                stderr=subprocess.STDOUT,
                            ),
                        )
                monitor(output, plan, processes)
            finally:
                # Also clean up if a later child fails to start, before releasing the run lock.
                for process in processes:
                    if process.poll() is None:
                        process.send_signal(signal.SIGINT)
                for process in processes:
                    process.wait()
            complete = load_chunks(output / "chunks", scores)
            if complete != total:
                message = "Scoring ended without complete contiguous protocol coverage"
                raise RuntimeError(message)
        atomic_json(output / "progress.json", {"status": "computing_metrics", "processed": total, "total": total})
        final_report(read_trials(Path(original.data.root), "eval"), scores, original, output)
        log.info("Completed parallel evaluation: %s", output)


def worker(cfg: DictConfig, output: Path) -> None:
    """Delegate one frozen shard to the existing score engine.

    Args:
        cfg: Worker preset with an explicit assignment index.
        output: Original evaluation directory; coordinator owns its run lock.
    """
    from src.steering.parallel_eval import worker as score_worker

    score_worker(output, int(cfg.evaluation.worker_index))
