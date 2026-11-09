"""Resumable full-protocol evaluation of frozen adaptation methods."""

import csv
import hashlib
import json
import logging
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, override

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_file
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from transformers import set_seed

from src.model.model import Wav2Vec2AASIST
from src.steering.evaluation import attack_report
from src.steering.finetune import build_model
from src.steering.inference import load_audio, run_model
from src.steering.midsteer import RankOneSteering
from src.steering.protocol import Trial, check_disjoint, read_trials, speaker_partition
from src.steering.run import apply_intervention, write_json
from src.steering.sae import TopKSAE

log = logging.getLogger(__name__)


class EvalAudio(Dataset[tuple[int, Tensor]]):
    """Decode full-protocol recordings lazily, without sampling or deduplication.

    Args:
        trials: All official eval records, in protocol order.
        num_samples: Fixed baseline crop/pad length.
        start: First unscored protocol index when resuming.
    """

    def __init__(self, trials: list[Trial], num_samples: int, start: int = 0) -> None:
        self.trials, self.num_samples, self.start = trials, num_samples, start

    def __len__(self) -> int:
        """Return the number of remaining recordings."""
        return len(self.trials) - self.start

    @override
    def __getitem__(self, index: int) -> tuple[int, Tensor]:
        """Decode one recording using unchanged deterministic preprocessing.

        Args:
            index: Relative index in the remaining protocol suffix.

        Returns:
            Absolute protocol index and CPU waveform; audio errors propagate.
        """
        absolute = self.start + index
        return absolute, load_audio([self.trials[absolute]], self.num_samples)[0]


@dataclass
class Scorer:
    """Frozen model and already selected intervention.

    Args:
        model: Detector in evaluation mode.
        layer: One-based transformer block.
        transform: Optional label-independent token correction.
    """

    model: Wav2Vec2AASIST
    layer: int
    transform: Callable[[Tensor], Tensor] | None = None

    def score(self, waves: Tensor, batch_size: int) -> np.ndarray:
        """Return bonafide probabilities for preprocessed audio.

        Args:
            waves: CPU waveform batch.
            batch_size: Forward microbatch size.

        Returns:
            Bonafide-positive probabilities in input order.
        """
        block = self.model.frontend.model.encoder.layers[self.layer - 1]
        return run_model(self.model, waves, block, batch_size, self.transform)[0]


def build_scorers(cfg: DictConfig) -> tuple[dict[str, Scorer], dict[str, str]]:
    """Restore saved selections; never fit or select on the full evaluation set.

    Args:
        cfg: Full-evaluation configuration with existing reference directories.

    Returns:
        Unique inference jobs and exact-zero fallback aliases.
    """
    settings = cfg.steering
    reference, fine_run, mid_run = map(Path, (settings.reference_run, settings.finetune_run, settings.midsteer_run))
    selected = json.loads((reference / "selection.json").read_text())["selected"]
    mid_selected = json.loads((mid_run / "selection.json").read_text())["selected"]
    fine_selected = json.loads((fine_run / "selection.json").read_text())
    model = build_model(cfg).requires_grad_(False).eval()
    sae_row = selected["sae"]
    layer = int(sae_row["layer"])
    weights = load_file(reference / f"sae_layer_{layer}.safetensors")
    sae = TopKSAE(len(weights["center"]), int(settings.expansion), int(settings.k)).to(str(settings.device))
    sae.load_state_dict(weights, strict=True)
    sae.requires_grad_(False).eval()
    directions = load_file(reference / f"directions_layer_{layer}.safetensors", device=str(settings.device))
    signs = directions["sae_direction"]
    order = torch.randperm(len(signs), generator=torch.Generator().manual_seed(int(cfg.seed))).to(signs.device)
    jobs = {
        "baseline": Scorer(model, layer),
        "sae": Scorer(model, layer, partial(sae.steer, direction=signs, strength=float(sae_row["strength"]))),
        "reconstruction": Scorer(model, layer, sae),
        "random_sae": Scorer(
            model,
            layer,
            partial(sae.steer, direction=signs[order], strength=float(sae_row["strength"])),
        ),
    }
    aliases = {}
    dense = selected["dense"]
    if dense["strength"] == 0:
        aliases["dense"] = "baseline"
    else:
        dense_vector = load_file(
            reference / f"directions_layer_{dense['layer']}.safetensors",
            device=str(settings.device),
        )["dense_direction"]
        jobs["dense"] = Scorer(
            model,
            int(dense["layer"]),
            partial(
                apply_intervention,
                method="dense",
                strength=float(dense["strength"]),
                fitted=(sae, signs, dense_vector),
            ),
        )
    fine = build_model(cfg).requires_grad_(False).eval()
    fine.load_state_dict(load_file(fine_run / "best_a21.safetensors"), strict=True)
    jobs["finetune_a21"] = Scorer(fine, layer)
    if fine_selected["guarded_epoch"] == 0:
        aliases["finetune_guarded"] = "baseline"
    elif fine_selected["guarded_epoch"] == fine_selected["best_a21_epoch"]:
        aliases["finetune_guarded"] = "finetune_a21"
    else:
        guarded = build_model(cfg).requires_grad_(False).eval()
        guarded.load_state_dict(load_file(fine_run / "guarded.safetensors"), strict=True)
        jobs["finetune_guarded"] = Scorer(guarded, layer)
    for method, row in mid_selected.items():
        state = load_file(mid_run / f"{method}_layer_{row['layer']}.safetensors", device=str(settings.device))
        adapter = RankOneSteering(state["anchor"], state["read"], state["write"])
        jobs[method] = Scorer(
            model,
            int(row["layer"]),
            partial(adapter.steer, strength=float(row["strength"]), gated=method.endswith("gated")),
        )
    methods = set(map(str, settings.methods))
    if methods != set(jobs) | set(aliases):
        message = "Full comparison must include every saved method; configuration and artifacts differ"
        raise ValueError(message)
    return jobs, aliases


def score_batch(
    jobs: dict[str, Scorer],
    aliases: dict[str, str],
    waves: Tensor,
    batch_size: int,
) -> dict[str, np.ndarray]:
    """Score each unique model once and reuse mathematically identical fallbacks.

    Args:
        jobs: Frozen forward jobs.
        aliases: Exact identity mappings, verified from saved selections.
        waves: Deterministically decoded input audio.
        batch_size: Forward batch size.

    Returns:
        Aligned score arrays for every comparison method.
    """
    scores = {name: job.score(waves, batch_size) for name, job in jobs.items()}
    scores.update({name: scores[target] for name, target in aliases.items()})
    return scores


def smoke_check(jobs: dict[str, Scorer], aliases: dict[str, str], cfg: DictConfig) -> dict[str, Any]:
    """Verify restored methods and larger batches against saved pilot scores.

    Args:
        jobs: Restored unique inference jobs.
        aliases: Selected exact-zero fallbacks.
        cfg: Smoke size and probability tolerance.

    Returns:
        Per-method maximum score difference and approximate combined throughput.
    """
    reference = Path(cfg.steering.midsteer_run)
    manifest = json.loads((reference / "manifest.json").read_text())["test"]
    saved = json.loads((reference / "scores.json").read_text())
    counts: Counter[str] = Counter()
    indices = []
    for index, row in enumerate(manifest):
        if counts[row["attack"]] < int(cfg.steering.smoke_per_attack):
            indices.append(index)
            counts[row["attack"]] += 1
    trials = [Trial(**manifest[index]) for index in indices]
    waves = load_audio(trials, int(cfg.data.num_samples))
    started = time.perf_counter()
    scores = score_batch(jobs, aliases, waves, int(cfg.steering.batch_size))
    elapsed = time.perf_counter() - started
    differences = {}
    for name, values in scores.items():
        expected = np.asarray(saved["scores"][name])[indices]
        np.testing.assert_allclose(
            values,
            expected,
            atol=float(cfg.steering.smoke_score_atol),
            rtol=0,
            err_msg=f"Restored method: {name}",
        )
        differences[name] = float(np.max(np.abs(values - expected)))
    return {"recordings": len(waves), "forward_seconds_all_methods": elapsed, "max_score_difference": differences}


def save_chunk(directory: Path, start: int, scores: dict[str, np.ndarray]) -> int:
    """Atomically save one contiguous chunk of finite scores.

    Args:
        directory: Run-owned chunk directory.
        start: First protocol index.
        scores: Equally sized per-method scores for the chunk.

    Returns:
        Exclusive ending protocol index.
    """
    lengths = {len(values) for values in scores.values()}
    if len(lengths) != 1 or not all(np.isfinite(values).all() for values in scores.values()):
        message = "Chunk scores must be finite and aligned across methods"
        raise ValueError(message)
    end = start + lengths.pop()
    temporary = directory / f"{start:09d}_{end:09d}.partial.npz"
    payload: dict[str, Any] = {"start": np.array(start), "end": np.array(end), **scores}
    np.savez_compressed(temporary, **payload)
    temporary.replace(directory / f"{start:09d}_{end:09d}.npz")
    return end


def load_chunks(directory: Path, scores: dict[str, np.ndarray]) -> int:
    """Validate and restore only a contiguous prefix of completed score chunks.

    Args:
        directory: Run-owned chunk directory.
        scores: Full-length destination arrays, initially NaN.

    Returns:
        First unscored protocol index. Partial temporary files are ignored.
    """
    expected = 0
    for path in sorted(directory.glob("*.npz")):
        if path.name.endswith(".partial.npz"):
            continue
        with np.load(path, allow_pickle=False) as chunk:
            start, end = int(chunk["start"]), int(chunk["end"])
            if (
                start != expected
                or not start < end <= len(next(iter(scores.values())))
                or set(chunk.files) != {"start", "end", *scores}
            ):
                message = f"Invalid or noncontiguous checkpoint chunk: {path}"
                raise ValueError(message)
            for method, destination in scores.items():
                values = chunk[method]
                if values.shape != (end - start,) or not np.isfinite(values).all():
                    message = f"Invalid score vector in {path}: {method}"
                    raise ValueError(message)
                destination[start:end] = values
            expected = end
    return expected


def stream_scores(
    trials: list[Trial],
    jobs: dict[str, Scorer],
    aliases: dict[str, str],
    cfg: DictConfig,
    output: Path,
) -> dict[str, np.ndarray]:
    """Stream full eval audio once, saving resumable score chunks without subsampling.

    Args:
        trials: Entire official evaluation protocol.
        jobs: Frozen unique inference methods.
        aliases: Exact baseline fallback aliases.
        cfg: Loader and checkpoint budgets.
        output: Validated output directory.

    Returns:
        Complete per-method scores; incomplete runs never produce final reports.
    """
    settings = cfg.steering
    scores = {str(name): np.full(len(trials), np.nan, dtype=np.float32) for name in settings.methods}
    chunks = output / "chunks"
    chunks.mkdir(exist_ok=True)
    processed = load_chunks(chunks, scores)
    initial, checkpointed = processed, processed
    dataset = EvalAudio(trials, int(cfg.data.num_samples), processed)
    loader = DataLoader(
        dataset,
        batch_size=int(settings.batch_size),
        shuffle=False,
        num_workers=int(settings.audio_workers),
        pin_memory=str(settings.device).startswith("cuda"),
        multiprocessing_context="spawn" if settings.audio_workers else None,
    )
    started, last_log = time.perf_counter(), 0.0
    for indices, waves in loader:
        np.testing.assert_array_equal(indices.numpy(), np.arange(processed, processed + len(waves)))
        batch = score_batch(jobs, aliases, waves, int(settings.batch_size))
        end = processed + len(waves)
        for method, values in batch.items():
            scores[method][processed:end] = values
        processed = end
        elapsed = time.perf_counter() - started
        if processed - checkpointed >= int(settings.chunk_size) or processed == len(trials):
            checkpointed = save_chunk(
                chunks,
                checkpointed,
                {name: values[checkpointed:processed] for name, values in scores.items()},
            )
        if elapsed - last_log >= 30 or processed == len(trials):
            rate = (processed - initial) / elapsed
            progress = {
                "status": "scoring",
                "processed": processed,
                "checkpointed": checkpointed,
                "total": len(trials),
                "session_seconds": elapsed,
                "recordings_per_second_all_methods": rate,
                "estimated_remaining_seconds": (len(trials) - processed) / rate,
            }
            write_json(output / "progress.json", progress)
            log.info("Full eval progress: %s", progress)
            last_log = elapsed
    return scores


def final_report(trials: list[Trial], scores: dict[str, np.ndarray], cfg: DictConfig, output: Path) -> None:
    """Report full-protocol and speaker-disjoint EERs with shared bonafide references.

    Args:
        trials: Entire eval protocol, no codec deduplication.
        scores: Complete, aligned probabilities for all methods.
        cfg: Original partition seed and frozen threshold references.
        output: Run artifact directory.
    """
    threshold = float(json.loads((Path(cfg.steering.reference_run) / "selection.json").read_text())["threshold"])
    heldout = np.array([speaker_partition(trial.speaker, int(cfg.seed)) == "test" for trial in trials])
    results = {}
    for cohort, mask in (("full_eval", np.ones(len(trials), dtype=bool)), ("heldout_speakers", heldout)):
        subset = [trial for trial, keep in zip(trials, mask, strict=True) if keep]
        results[cohort] = {name: attack_report(subset, values[mask], threshold) for name, values in scores.items()}
    write_json(
        output / "report.json",
        {
            "threshold": threshold,
            "results": results,
            "interpretation": (
                "Single EER estimates, not bootstrap means. Full eval includes adaptation/tuning speakers; "
                "heldout_speakers excludes both reserved pools. Every attack uses all bonafide recordings "
                "in its cohort. All codecs retained."
            ),
        },
    )
    with (output / "comparison.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cohort", "attack", "method", "eer_percent", "min_dcf", "fixed_dcf", "n_bonafide", "n_spoof"])
        for cohort, methods in results.items():
            for method, attacks in methods.items():
                for attack, values in attacks.items():
                    writer.writerow(
                        [
                            cohort,
                            attack,
                            method,
                            100 * values["eer"],
                            values["min_dcf"],
                            values["fixed_dcf"],
                            int(values["n_bonafide"]),
                            int(values["n_spoof"]),
                        ],
                    )
    write_json(output / "progress.json", {"status": "complete", "processed": len(trials), "total": len(trials)})


@hydra.main(version_base="1.3", config_path="../../configs", config_name="main")
def main(cfg: DictConfig) -> None:
    """Restore frozen models, verify pilot equivalence, and score the full protocol.

    Args:
        cfg: Configuration using steering=full_eval; optionally resume_dir.
    """
    set_seed(int(cfg.seed))
    torch.set_num_threads(4)
    settings = cfg.steering
    output = Path(settings.resume_dir or cfg.paths.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    protocol = Path(cfg.data.root) / "last_eval/ASVspoof5.eval.track_1.tsv"
    reference, fine_run, mid_run = map(Path, (settings.reference_run, settings.finetune_run, settings.midsteer_run))
    original = json.loads((reference / "manifest.json").read_text())
    if any(json.loads((directory / "manifest.json").read_text()) != original for directory in (fine_run, mid_run)):
        message = "Reference adaptation manifests differ"
        raise ValueError(message)
    sources = [
        protocol,
        Path(settings.checkpoint) / "model.safetensors",
        *reference.glob("*.safetensors"),
        *fine_run.glob("*.safetensors"),
        *mid_run.glob("*.safetensors"),
        *Path("src/steering").glob("*.py"),
        Path("src/model/model.py"),
        Path("src/data/dataset.py"),
        Path("src/metrics/detection.py"),
    ]
    sources.extend(
        directory / name
        for directory in (reference, fine_run, mid_run)
        for name in ("selection.json", "manifest.json", "provenance.json")
    )
    hashes = {}
    for path in sources:
        with path.open("rb") as handle:
            hashes[str(path)] = hashlib.file_digest(handle, "sha256").hexdigest()
    base_digest = hashes[str(Path(settings.checkpoint) / "model.safetensors")]
    if base_digest != json.loads((reference / "provenance.json").read_text())["checkpoint_sha256"]:
        message = "Baseline checkpoint differs from original adaptation"
        raise ValueError(message)
    identity = {
        "sources_sha256": hashes,
        "seed": int(cfg.seed),
        "num_samples": int(cfg.data.num_samples),
        "batch_size": int(settings.batch_size),
        "methods": list(settings.methods),
        "precision": "FP32; no autocast",
    }
    manifest_path = output / "provenance.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text())["identity"] != identity:
        message = "Resume sources or scoring configuration differ; use a new run directory"
        raise ValueError(message)
    trials = read_trials(Path(cfg.data.root), "eval")
    partitions = {name: [Trial(**row) for row in rows] for name, rows in original.items() if name != "test"}
    partitions["heldout"] = [trial for trial in trials if speaker_partition(trial.speaker, int(cfg.seed)) == "test"]
    check_disjoint(partitions)
    counts = dict(sorted(Counter(trial.attack for trial in trials).items()))
    write_json(
        manifest_path,
        {
            "identity": identity,
            "counts": counts,
            "n_recordings": len(trials),
            "protocol_order": "Exact official eval TSV order; chunks index this order",
            "config": OmegaConf.to_container(cfg, resolve=True),
            "torch": torch.__version__,
        },
    )
    jobs, aliases = build_scorers(cfg)
    smoke = smoke_check(jobs, aliases, cfg)
    write_json(output / "smoke.json", smoke)
    log.info("Restored-score smoke passed: %s", smoke)
    if settings.smoke_only:
        log.info("Smoke only; full eval was not started")
        return
    log.info("Full eval counts: %s; outputs: %s", counts, output)
    scores = stream_scores(trials, jobs, aliases, cfg, output)
    final_report(trials, scores, cfg, output)
    log.info("Completed full evaluation: %s", output)


if __name__ == "__main__":
    main()
