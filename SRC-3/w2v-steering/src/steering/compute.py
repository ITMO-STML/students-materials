"""Synchronized timing and common-workload inference benchmarking."""

import time
from collections.abc import Callable
from typing import Any

import numpy as np
import torch


def measured[T](device: torch.device, operation: Callable[[], T]) -> tuple[T, dict[str, float]]:
    """Measure synchronized wall time and process tensor allocation.

    Args:
        device: Execution device to synchronize, if CUDA.
        operation: Synchronous workload, excluding setup done by the caller.

    Returns:
        Workload result and wall seconds / CUDA tensor-memory measurements.
        These are not GPU busy time, energy, or total device memory usage.
    """
    allocated = 0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        allocated = torch.cuda.memory_allocated(device)
    started = time.perf_counter()
    result = operation()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    return result, {
        "wall_seconds": seconds,
        "peak_allocated_mib": peak / 2**20,
        "extra_peak_allocated_mib": max(0, peak - allocated) / 2**20,
    }


def summarize_rounds(
    samples: dict[str, list[dict[str, float]]],
    clips: int,
    audio_seconds: float,
) -> dict[str, dict[str, float | int | None]]:
    """Calculate arithmetic means and sample standard deviations across rounds.

    Args:
        samples: Per-method rounds in matching repetition order, including baseline.
        clips: Number of clips processed in each round, not independent replicates.
        audio_seconds: Fixed duration per clip.

    Returns:
        Explicit mean/std fields with ddof=1. Std is None for a single round.
        Throughput is transformed per round before averaging; overhead is paired
        to the baseline measurement from the same randomized round.
    """
    if clips < 1 or audio_seconds <= 0 or not samples.get("baseline"):
        message = "Require positive clip count/duration and nonempty baseline rounds"
        raise ValueError(message)
    reference = np.array([row["wall_seconds"] for row in samples["baseline"]], dtype=float)
    if not np.isfinite(reference).all() or (reference <= 0).any():
        message = "Round durations must be finite and positive"
        raise ValueError(message)
    summary = {}
    for name, rows in samples.items():
        times = np.array([row["wall_seconds"] for row in rows], dtype=float)
        if len(times) != len(reference) or not np.isfinite(times).all() or (times <= 0).any():
            message = "Methods require aligned rounds with finite positive durations"
            raise ValueError(message)
        metrics = {
            "wall_seconds": times,
            "ms_per_clip": times * 1000 / clips,
            "clips_per_second": clips / times,
            "real_time_factor": times / (clips * audio_seconds),
            "paired_overhead_percent": 100 * (times / reference - 1),
            "extra_peak_allocated_mib": np.array([row["extra_peak_allocated_mib"] for row in rows]),
        }
        values: dict[str, float | int | None] = {"n_rounds": len(rows), "std_ddof": 1}
        for metric, array in metrics.items():
            values[f"mean_{metric}"] = float(array.mean())
            values[f"std_{metric}"] = float(array.std(ddof=1)) if len(array) > 1 else None
        summary[name] = values
    return summary


def benchmark(
    jobs: dict[str, Callable[[], Any]],
    device: torch.device,
    clips: int,
    audio_seconds: float,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    """Warm up and interleave methods in randomized order on identical audio.

    Args:
        jobs: Named inference callables; models and audio already loaded.
        device: Shared execution device.
        clips: Number of recordings scored by each callable.
        audio_seconds: Duration per fixed-length input.
        repeats: Number of independently ordered measurement rounds.
        seed: Randomized execution-order seed.

    Returns:
        Raw rounds, legacy median summaries, and explicit mean/sample-std fields.
    """
    if clips < 1 or repeats < 1 or audio_seconds <= 0 or "baseline" not in jobs:
        message = "Benchmark requires positive budgets and a baseline"
        raise ValueError(message)
    for operation in jobs.values():
        measured(device, operation)
    rng = np.random.default_rng(seed)
    samples: dict[str, list[dict[str, float]]] = {name: [] for name in jobs}
    orders = []
    for _ in range(repeats):
        order = rng.permutation(list(jobs)).tolist()
        orders.append(order)
        for name in order:
            _, timing = measured(device, jobs[name])
            samples[name].append(timing)
    reference = np.median([row["wall_seconds"] for row in samples["baseline"]])
    summary: dict[str, dict[str, float | int | None]] = {}
    for name, rows in samples.items():
        times = np.array([row["wall_seconds"] for row in rows])
        median = float(np.median(times))
        summary[name] = {
            "ms_per_clip": median * 1000 / clips,
            "clips_per_second": clips / median,
            "real_time_factor": median / (clips * audio_seconds),
            "overhead_percent": float(100 * (median / reference - 1)),
            "min_ms_per_clip": float(times.min()) * 1000 / clips,
            "max_ms_per_clip": float(times.max()) * 1000 / clips,
            "max_extra_peak_allocated_mib": max(row["extra_peak_allocated_mib"] for row in rows),
        }
    round_summary = summarize_rounds(samples, clips, audio_seconds)
    for name, values in summary.items():
        values.update(round_summary[name])
    return {"summary": summary, "samples": samples, "round_orders": orders}
