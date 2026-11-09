"""Frozen A21 mean transport, held-out comparison, and measured compute costs."""

import csv
import hashlib
import json
import logging
import re
import time
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from safetensors.torch import load_file, save_file
from torch import Tensor
from transformers import set_seed

from src.steering.compute import benchmark, measured
from src.steering.evaluation import attack_report
from src.steering.finetune import build_model
from src.steering.inference import load_audio, run_model
from src.steering.midsteer import RankOneSteering, fit_transport
from src.steering.protocol import Trial, check_disjoint
from src.steering.run import admissible, candidate_metrics, encoder_block, write_json
from src.steering.sae import TopKSAE, fit_sae

if TYPE_CHECKING:
    from collections.abc import Callable

log = logging.getLogger(__name__)
METHODS = ("midsteer_affine", "midsteer_anchored_gated")


def sampled_tokens(hidden: Tensor, count: int) -> Tensor:
    """Select equally spaced tokens, giving every recording the same weight.

    Args:
        hidden: Clip, time, channel activations.
        count: Maximum token count per clip.

    Returns:
        Clip, sampled time, channel tensor.
    """
    indices = torch.linspace(0, hidden.shape[1] - 1, min(count, hidden.shape[1])).long()
    return hidden[:, indices]


def fit_layer(
    cached: dict[str, Tensor],
    partitions: dict[str, list[Trial]],
    settings: DictConfig,
) -> dict[str, RankOneSteering]:
    """Fit both predeclared transport variants from training/support only.

    Args:
        cached: Original training/support block activations on CPU.
        partitions: Audited manifests.
        settings: Token, device, and ridge settings.

    Returns:
        Affine and bonafide-anchored adapters.
    """
    tokens = {name: sampled_tokens(hidden, int(settings.tokens_per_clip)) for name, hidden in cached.items()}
    train_spoof = torch.tensor([trial.label == 1 for trial in partitions["sae_train"]])
    support_spoof = torch.tensor([trial.label == 1 for trial in partitions["support"]])
    bona = torch.cat((tokens["sae_train"][~train_spoof], tokens["support"][~support_spoof]))
    groups = tuple(
        value.flatten(0, 1).to(str(settings.device))
        for value in (bona, tokens["sae_train"][train_spoof], tokens["support"][support_spoof])
    )
    # Explicit tuple preserves the fixed three-group type for static checking.
    inputs = (groups[0], groups[1], groups[2])
    return {
        method: fit_transport(inputs, float(settings.ridge), anchored=method.endswith("gated")) for method in METHODS
    }


def historical_timing(directory: Path, marker: str) -> dict[str, Any]:
    """Extract coarse historical wall intervals without inventing fit timings.

    Args:
        directory: Existing Hydra run named with start time.
        marker: Log message identifying the end of adaptation/selection.

    Returns:
        Log-derived intervals including startup, audio, fitting, and validation.
    """
    # A shared fixed offset suffices for intervals; logs do not record a zone.
    started = datetime.strptime(f"{directory.parent.name} {directory.name[-8:]}", "%Y-%m-%d %H-%M-%S").replace(
        tzinfo=UTC,
    )
    logs = list(directory.glob("*.log"))
    events = []
    for path in logs:
        for line in path.read_text().splitlines():
            match = re.match(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3})\]", line)
            if match:
                events.append((datetime.strptime(match[1], "%Y-%m-%d %H:%M:%S,%f").replace(tzinfo=UTC), line))
    selected = next(timestamp for timestamp, line in events if marker in line)
    return {
        "run": str(directory),
        "start_to_selection_seconds": (selected - started).total_seconds(),
        "start_to_last_log_seconds": (max(timestamp for timestamp, _ in events) - started).total_seconds(),
        "scope": "Historical wall time incl. startup/audio/fit/tuning/checkpoint I/O; final test only in total.",
        "caveat": "Different run times and shared GPU loads; not a controlled training-speed comparison.",
    }


@hydra.main(version_base="1.3", config_path="../../configs", config_name="main")
def main(cfg: DictConfig) -> None:  # noqa: C901 -- Sequential experiment phases keep test access visibly after selection.
    """Run closed-form adaptation and benchmark already selected methods.

    Args:
        cfg: Configuration using steering=midsteer_a21.
    """
    started = time.perf_counter()
    set_seed(int(cfg.seed))
    torch.set_num_threads(4)
    settings = cfg.steering
    device = torch.device(str(settings.device))
    output, reference, fine_run = Path(cfg.paths.output_dir), Path(settings.reference_run), Path(settings.finetune_run)
    output.mkdir(parents=True, exist_ok=True)
    raw = json.loads((reference / "manifest.json").read_text())
    if raw != json.loads((fine_run / "manifest.json").read_text()):
        message = "Comparator manifests differ"
        raise ValueError(message)
    partitions = {name: [Trial(**row) for row in rows] for name, rows in raw.items()}
    check_disjoint(partitions)
    if {trial.attack for trial in partitions["support"]} != {"A21", "bonafide"}:
        message = "Only labeled A21 and bonafide support are permitted"
        raise ValueError(message)
    write_json(output / "manifest.json", raw)
    previous = json.loads((reference / "selection.json").read_text())
    threshold = float(previous["threshold"])
    with (Path(settings.checkpoint) / "model.safetensors").open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if digest != json.loads((reference / "provenance.json").read_text())["checkpoint_sha256"]:
        message = "Reference detector weights differ"
        raise ValueError(message)
    provenance = {
        "checkpoint_sha256": digest,
        "reference_run": str(reference),
        "finetune_run": str(fine_run),
        "upstream_commit": "0f3b31e15cdda6ad0d46167e10319e896d6f1541",
        "method": "Independent regularized rank-one MidSteer-inspired A21-to-known-spoof transport",
        "selection": "Predeclared layer/strength sweep; original A21 tune/dev guards; no transfer selection",
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in Path("src/steering").glob("*.py")
        },
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "torch": torch.__version__,
        "seed": int(cfg.seed),
    }
    write_json(output / "provenance.json", provenance)
    timings: dict[str, Any] = {}
    model, timings["model_load"] = measured(device, lambda: build_model(cfg).requires_grad_(False))
    waves, timings["audio_load_non_test"] = measured(
        device,
        lambda: {
            name: load_audio(partitions[name], int(cfg.data.num_samples))
            for name in ("sae_train", "support", "tune", "dev")
        },
    )
    base_scores, timings["baseline_calibration_check"] = measured(
        device,
        lambda: {
            name: run_model(model, waves[name], encoder_block(model, 6), int(settings.batch_size))[0]
            for name in ("tune", "dev")
        },
    )
    original_scores = json.loads((reference / "baseline_calibration_scores.json").read_text())
    for name in base_scores:
        np.testing.assert_allclose(base_scores[name], original_scores[name]["scores"], atol=1e-5, rtol=1e-5)
    baseline = candidate_metrics(partitions, base_scores, threshold)
    fitted = {}
    candidates: list[dict[str, Any]] = []
    sae_replay: dict[str, Any] = {}
    for layer in map(int, settings.layers):
        block = encoder_block(model, layer)
        captured, timings[f"extract_layer_{layer}"] = measured(
            device,
            lambda block=block: {
                name: run_model(model, waves[name], block, int(settings.batch_size), capture=True)[1]
                for name in ("sae_train", "support")
            },
        )
        assert all(hidden is not None for hidden in captured.values())
        cached = {name: hidden for name, hidden in captured.items() if hidden is not None}
        fitted[layer], timings[f"fit_layer_{layer}"] = measured(
            device,
            partial(fit_layer, cached, partitions, settings),
        )
        for method, adapter in fitted[layer].items():
            save_file(
                {key: value.cpu().contiguous() for key, value in adapter.state_dict().items()},
                output / f"{method}_layer_{layer}.safetensors",
            )
        log.info(
            "Layer %s: extraction %.3fs; both closed-form fits %.3fs",
            layer,
            timings[f"extract_layer_{layer}"]["wall_seconds"],
            timings[f"fit_layer_{layer}"]["wall_seconds"],
        )
        if bool(settings.benchmark_sae_fit) and layer == int(previous["selected"]["sae"]["layer"]):
            tokens = sampled_tokens(cached["sae_train"], int(settings.tokens_per_clip)).flatten(0, 1)
            replay = TopKSAE(tokens.shape[-1], int(settings.expansion), int(settings.k)).to(device)
            _, replay_timing = measured(
                device,
                partial(fit_sae, replay, tokens, int(settings.sae_steps), int(settings.sae_batch_size), int(cfg.seed)),
            )
            sae_replay = {
                **replay_timing,
                "layer": layer,
                "steps": int(settings.sae_steps),
                "tokens": len(tokens),
                "scope": (
                    "One-layer SAE optimizer replay on cached tokens; excludes extraction, probe, validation; "
                    "original selected weights retained."
                ),
            }
            del replay, tokens
        del captured, cached
        for method in METHODS:
            for strength in sorted({0.0, *map(float, settings.strengths)}):
                if strength == 0:
                    metrics = dict(baseline)
                else:
                    transform = partial(fitted[layer][method].steer, strength=strength, gated=method.endswith("gated"))
                    scores, timing = measured(
                        device,
                        lambda block=block, transform=transform: {
                            name: run_model(model, waves[name], block, int(settings.batch_size), transform)[0]
                            for name in ("tune", "dev")
                        },
                    )
                    timings[f"select_{layer}_{method}_{strength}"] = timing
                    metrics = candidate_metrics(partitions, scores, threshold)
                row: dict[str, Any] = {"method": method, "layer": layer, "strength": strength, **metrics}
                row["admissible"] = admissible(row, baseline, settings)
                candidates.append(row)
                log.info("Candidate: %s", row)
    selected = {
        method: min(
            (row for row in candidates if row["method"] == method and row["admissible"]),
            key=lambda row: (row["tune_eer"], row["tune_min_dcf"], abs(row["strength"]), row["layer"]),
        )
        for method in METHODS
    }
    write_json(
        output / "selection.json",
        {"threshold": threshold, "baseline": baseline, "selected": selected, "candidates": candidates},
    )
    adaptation_wall = time.perf_counter() - started - sae_replay.get("wall_seconds", 0)
    log.info("Selection fixed before test access: %s", selected)
    test_waves, timings["audio_load_test"] = measured(
        device,
        lambda: load_audio(partitions["test"], int(cfg.data.num_samples)),
    )
    reports = json.loads((fine_run / "report.json").read_text())["results"]
    saved_scores = json.loads((fine_run / "scores.json").read_text())
    if saved_scores["utterances"] != [trial.utterance for trial in partitions["test"]]:
        message = "Reference score order differs from manifest"
        raise ValueError(message)
    for method in METHODS:
        row = selected[method]
        transform = partial(
            fitted[row["layer"]][method].steer,
            strength=float(row["strength"]),
            gated=method.endswith("gated"),
        )
        result, timings[f"test_{method}"] = measured(
            device,
            partial(
                run_model,
                model,
                test_waves,
                encoder_block(model, row["layer"]),
                int(settings.batch_size),
                transform,
            ),
        )
        scores = result[0]
        saved_scores["scores"][method] = scores.tolist()
        reports[method] = attack_report(partitions["test"], scores, threshold)
        for attack, values in reports[method].items():
            base = reports["baseline"][attack]
            values.update(
                delta_eer_pp=100 * (values["eer"] - base["eer"]),
                delta_min_dcf=values["min_dcf"] - base["min_dcf"],
                delta_fixed_dcf=values["fixed_dcf"] - base["fixed_dcf"],
            )
        log.info(
            "%s held-out: %s",
            method,
            {attack: reports[method][attack] for attack in ("pooled", "A21", "A22", "A31")},
        )
    write_json(
        output / "report.json",
        {
            "results": reports,
            "threshold": threshold,
            "selection": selected,
            "interpretation": (
                "Single-seed exploratory reuse of the existing test cohort; no test-directed tuning within this run."
            ),
        },
    )
    write_json(output / "scores.json", saved_scores)
    with (output / "comparison.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["attack", "method", "eer_percent", "delta_eer_pp", "min_dcf", "fixed_dcf", "bonafide_rejection_percent"],
        )
        for attack in reports["baseline"]:
            for method in reports:
                row = reports[method][attack]
                writer.writerow(
                    [
                        attack,
                        method,
                        100 * row["eer"],
                        100 * (row["eer"] - reports["baseline"][attack]["eer"]),
                        row["min_dcf"],
                        row["fixed_dcf"],
                        100 * row["bonafide_rejection"],
                    ],
                )

    # No benchmark result feeds back into model selection or held-out metrics.
    log.info("Benchmarking common-workload inference, including the saved A21-trained model")
    sae_row = previous["selected"]["sae"]
    sae_file = reference / f"sae_layer_{sae_row['layer']}.safetensors"
    sae_weights = load_file(sae_file)
    width = len(sae_weights["center"])
    sae = TopKSAE(width, int(settings.expansion), int(settings.k)).to(device)
    sae.load_state_dict(sae_weights, strict=True)
    sae.requires_grad_(False).eval()
    signs = load_file(reference / f"directions_layer_{sae_row['layer']}.safetensors")["sae_direction"].to(device)
    fine = build_model(cfg).requires_grad_(False)
    fine.load_state_dict(load_file(fine_run / "best_a21.safetensors"), strict=True)
    benchmark_waves = waves["tune"][: int(settings.benchmark_clips)]
    batch_size = int(settings.batch_size)
    jobs: dict[str, Callable[[], Any]] = {
        "baseline": partial(run_model, model, benchmark_waves, encoder_block(model, 6), batch_size),
        "sae": partial(
            run_model,
            model,
            benchmark_waves,
            encoder_block(model, int(sae_row["layer"])),
            batch_size,
            partial(sae.steer, direction=signs, strength=float(sae_row["strength"])),
        ),
        "finetune_a21": partial(run_model, fine, benchmark_waves, encoder_block(fine, 6), batch_size),
    }
    for method, row in selected.items():
        jobs[method] = partial(
            run_model,
            model,
            benchmark_waves,
            encoder_block(model, row["layer"]),
            batch_size,
            partial(
                fitted[row["layer"]][method].steer,
                strength=float(row["strength"]),
                gated=method.endswith("gated"),
            ),
        )
        # If selection fell back to zero, still expose the actual nonzero adapter cost.
        if row["strength"] == 0:
            jobs[f"{method}_nonzero_cost_only"] = partial(
                run_model,
                model,
                benchmark_waves,
                encoder_block(model, row["layer"]),
                batch_size,
                partial(fitted[row["layer"]][method].steer, strength=1.0, gated=method.endswith("gated")),
            )
    latency = benchmark(
        jobs,
        device,
        len(benchmark_waves),
        int(cfg.data.num_samples) / 16000,
        int(settings.benchmark_repeats),
        int(cfg.seed),
    )
    costs = {
        "baseline_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "sae_trainable_parameters": sum(parameter.numel() for parameter in sae.parameters()),
        "sae_selected_artifact_bytes": sae_file.stat().st_size,
        "sae_extra_linear_macs_per_token": 2 * width * width * int(settings.expansion),
        "midsteer_fitted_scalars": 3 * width,
        "midsteer_selected_artifact_bytes": {
            method: (output / f"{method}_layer_{row['layer']}.safetensors").stat().st_size
            for method, row in selected.items()
        },
        "midsteer_extra_dot_and_scale_operations_per_token": 2 * width,
        "operation_note": (
            "Analytical adapter-only dominant operations, not measured full-model FLOPs; "
            "excludes centering/additions/TopK. Rank-one dot and scaling vs SAE two dense projections."
        ),
        "finetune_checkpoint_bytes": (fine_run / "best_a21.safetensors").stat().st_size,
        "optimizer_steps": {"midsteer": 0, "sae_per_layer": int(settings.sae_steps), "finetune": 80},
    }
    compute = {
        "device_name": provenance["device_name"],
        "inference_precision": "FP32, eval, no autocast",
        "inference_scope": (
            "Preloaded CPU audio -> device transfer -> forward with temporary hook -> CPU probabilities; "
            "excludes audio/checkpoint I/O. All methods resident; peak increments exclude resident models. "
            "Shared-server timing, not energy or GPU busy time."
        ),
        "batch_size": batch_size,
        "clips_per_repeat": len(benchmark_waves),
        "samples_per_clip": int(cfg.data.num_samples),
        "inference": latency,
        "adaptation_stages": timings,
        "midsteer_start_to_selection_seconds_excluding_sae_replay": adaptation_wall,
        "midsteer_closed_form_fit_seconds_both_variants_all_layers": sum(
            value["wall_seconds"] for key, value in timings.items() if key.startswith("fit_layer_")
        ),
        "sae_one_layer_fit_replay": sae_replay,
        "historical_sae": historical_timing(reference, "Selection fixed; now scoring"),
        "historical_finetune": historical_timing(fine_run, "Supervised checkpoint selection fixed"),
        "costs": costs,
    }
    write_json(output / "compute.json", compute)
    with (output / "compute.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", *next(iter(latency["summary"].values())).keys()])
        for method, values in latency["summary"].items():
            writer.writerow([method, *values.values()])
    log.info("Inference benchmark: %s", latency["summary"])
    log.info("Completed MidSteer comparison: %s", output)


if __name__ == "__main__":
    main()
