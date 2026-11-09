"""Repeated cached-audio scoring latency for representative ALM interventions."""

from __future__ import annotations

import time
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from src.alm_steering.interventions import Intervention, steer

if TYPE_CHECKING:
    from torch import Tensor

    from src.alm_steering.runtime import FrozenALM


def benchmark(model: FrozenALM, audio: Tensor, directions: dict[str, dict[str, Tensor]], layer: int) -> dict[str, Any]:
    """Measure five synchronized rounds after a warmup, without audio encoding.

    Args:
        model: Frozen baseline ALM.
        audio: One fixed batch of tuning audio embeddings.
        directions: Previously fitted intervention vectors.
        layer: One-based representative block, fixed before test evaluation.

    Returns:
        Per-round wall times, mean/sample standard deviation and peak GPU memory.
    """
    with torch.no_grad():
        prefix, span = model.prefix(audio)
    cases = [("baseline", None, "audio")]
    cases += [(f"text_mean_{scope}", f"text_mean_base_{layer}", scope) for scope in ("audio", "text", "both")]
    cases += [("audio_transport", f"audio_transport_{layer}", "audio")]
    results = {}
    for name, direction, scope in cases:
        context = nullcontext()
        if direction is not None:
            intervention = Intervention(
                **directions[direction],
                strength=1.0,
                scope=scope,
                prefix_length=prefix.shape[1],
                audio_span=span,
            )
            context = steer(model.layers[layer - 1], intervention)
        with context:
            model.score(audio)
            torch.cuda.synchronize(model.device)
            torch.cuda.reset_peak_memory_stats(model.device)
            rounds = []
            for _ in range(5):
                started = time.perf_counter()
                model.score(audio)
                torch.cuda.synchronize(model.device)
                rounds.append(time.perf_counter() - started)
        results[name] = {
            "seconds": rounds,
            "mean_seconds": float(np.mean(rounds)),
            "std_seconds": float(np.std(rounds, ddof=1)),
            "peak_allocated_bytes_including_model": torch.cuda.max_memory_allocated(model.device),
        }
    return {
        "scope": "cached projected audio to two-answer likelihood scores; excludes audio encoder and disk IO",
        "batch_size": len(audio),
        "layer": layer,
        "strength": 1.0,
        "warmup_rounds": 1,
        "measured_rounds": 5,
        "results": results,
    }
