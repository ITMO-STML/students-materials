"""Resumable native SASV baseline and frozen steering pilot, with no training."""

import argparse
import fcntl
import json
import logging
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor

from src.alm_reasoning.artifacts import digest, write_json
from src.alm_sasv.metrics import parse_answer, report
from src.alm_sasv.protocol import prepare
from src.alm_sasv.runtime import SASVALM
from src.alm_sasv.smoke import smoke
from src.alm_sasv.steering import allowed, fit_directions, prefill_steer
from src.alm_steering.interventions import Intervention

LOGGER = logging.getLogger(__name__)
FAMILIES = ("audio_mean", "transport", "text_mean", "text_probe")


def save_tensor(path: Path, value: Any) -> None:
    """Atomically save run-owned tensors.

    Args:
        path: Destination.
        value: Tensor tree.
    """
    temporary = path.with_suffix(".partial")
    torch.save(value, temporary)
    temporary.replace(path)


def encode(model: SASVALM, rows: list[dict[str, Any]]) -> Tensor:
    """Encode a small pair group using the fixed batch size.

    Args:
        model: Frozen runtime.
        rows: Ordered pair records.

    Returns:
        CPU projected audio tokens.
    """
    size = model.cfg["batch_size"]
    return torch.cat([model.encode_pairs(rows[i : i + size]) for i in range(0, len(rows), size)])


def evaluate_group(
    model: SASVALM,
    rows: list[dict[str, Any]],
    name: str,
    settings: dict[str, Any],
    directions: dict[str, Any],
    output: Path,
    cached: Tensor | None = None,
) -> dict[str, list[int]]:
    """Score all settings per audio chunk, reusing encoding and saved predictions.

    Args:
        model: Frozen SASV runtime.
        rows: Exact ordered pair cohort.
        name: Unique cohort name.
        settings: Named layer/strength settings; None denotes baseline.
        directions: Immutable fitted vectors.
        output: Experiment directory.
        cached: Optional small-cohort acoustic cache.

    Returns:
        Ordered parsed predictions for each setting.
    """
    predictions: dict[str, list[int]] = {key: [] for key in settings}
    totals = dict.fromkeys(settings, 0.0)
    encoding_seconds = 0.0
    directory = output / "predictions" / name
    directory.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(rows), model.cfg["chunk_size"]):
        batch_rows = rows[start : start + model.cfg["chunk_size"]]
        destination = directory / f"{start:09d}.json"
        if destination.exists():
            chunk = json.loads(destination.read_text())
            if chunk["ids"] != [row["id"] for row in batch_rows] or chunk["settings"] != settings:
                message = f"Immutable pair chunk identity mismatch: {destination}"
                raise ValueError(message)
        else:
            started = time.perf_counter()
            audio = cached[start : start + len(batch_rows)] if cached is not None else encode(model, batch_rows)
            chunk = {
                "ids": [row["id"] for row in batch_rows],
                "settings": settings,
                "encoding_seconds": time.perf_counter() - started,
                "methods": {},
            }
            for method, setting in settings.items():
                started = time.perf_counter()
                generated = []
                torch.cuda.reset_peak_memory_stats(model.device)
                for index in range(0, len(audio), model.cfg["batch_size"]):
                    batch = audio[index : index + model.cfg["batch_size"]]
                    context = nullcontext()
                    if setting:
                        prefix, span = model.prefix(batch)
                        intervention = Intervention(
                            strength=setting["strength"],
                            scope="audio",
                            prefix_length=prefix.shape[1],
                            audio_span=span,
                            **directions[setting["direction"]],
                        )
                        context = prefill_steer(model.layers[setting["layer"] - 1], intervention)
                    with context:
                        generated.extend(model.decode(batch))
                torch.cuda.synchronize(model.device)
                chunk["methods"][method] = {
                    "wall_seconds": time.perf_counter() - started,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(model.device),
                    "predictions": [{**row, "prediction": parse_answer(row["text"])} for row in generated],
                }
                LOGGER.info("%s %s: %d/%d pairs", name, method, start + len(batch_rows), len(rows))
            write_json(destination, chunk)
        encoding_seconds += chunk["encoding_seconds"]
        for method, result in chunk["methods"].items():
            for row in result["predictions"]:
                if parse_answer(row["text"]) != row["prediction"]:
                    message = "Saved SASV prediction no longer matches its text"
                    raise ValueError(message)
                predictions[method].append(row["prediction"])
            totals[method] += result["wall_seconds"]
        write_json(
            output / "progress.json",
            {
                "status": "running",
                "stage": name,
                "completed": start + len(batch_rows),
                "total": len(rows),
                "settings": list(settings),
                "generation_seconds": totals,
                "encoding_seconds": encoding_seconds,
            },
        )
    metrics = {key: report(rows, values) for key, values in predictions.items()}
    write_json(
        output / f"{name}_report.json",
        {
            "status": "complete",
            "cohort": name,
            "n": len(rows),
            "metrics": metrics,
            "generation_seconds": totals,
            "encoding_seconds": encoding_seconds,
            "note": "Native pair cohort; not single-recording or full official evaluation.",
        },
    )
    return predictions


def rank(metrics: dict[str, Any], setting: dict[str, Any] | None) -> tuple[float, ...]:
    """Rank on binary tuning F1, then native macro F1 and accuracy.

    Args:
        metrics: Complete tuning-group metrics.
        setting: Candidate or zero-steering baseline.

    Returns:
        Ascending key; exact ties prefer zero steering.
    """
    return (
        -metrics["binary"]["f1"],
        -metrics["macro_f1"],
        -metrics["accuracy"],
        abs(setting["strength"]) if setting else 0.0,
        setting["layer"] if setting else 0.0,
    )


def candidate_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    """Create the prespecified steering grid, including the zero fallback.

    Args:
        cfg: Fixed layers and strengths.

    Returns:
        Named candidate settings.
    """
    candidates: dict[str, Any] = {"baseline": None}
    for family in FAMILIES:
        for layer in cfg["layers"]:
            for strength in cfg["strengths"]:
                candidates[f"{family}_{layer}_{strength:+g}"] = {
                    "family": family,
                    "direction": f"{family}_{layer}",
                    "layer": layer,
                    "strength": strength,
                }
    return candidates


def run(cfg: dict[str, Any]) -> None:
    """Execute a resumable baseline, guarded tuning, and paired transfer pilot.

    Args:
        cfg: Fixed experiment configuration.
    """
    output = Path(cfg["output"])
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(4)
        torch.manual_seed(cfg["seed"])
        root, pairs, source = Path(cfg["dataset_root"]), Path(cfg["pairs_root"]), Path(cfg["source_root"])
        inputs = [
            Path(cfg["checkpoint"]),
            Path(cfg["checkpoint_config"]),
            Path(cfg["test_config"]),
            *sorted(Path("src/alm_sasv").glob("*.py")),
            *sorted(Path("src/alm_reasoning").glob("*.py")),
            *sorted(Path("src/alm_steering").glob("*.py")),
            Path("src/steering/midsteer.py"),
            Path("src/steering/protocol.py"),
            *sorted((source / "src").rglob("*.py")),
            source / "src/prompts/sasv_reasoning_prompts.json",
            *[
                pairs / name
                for name in (
                    "sasv_asv5_dev_pairs_balanced.json",
                    "sasv_asv5_eval_pairs.json",
                    "sasv_asv5_eval_pairs_20K.json",
                )
            ],
            *[
                root / name
                for name in (
                    "ASVspoof5.dev.metadata.txt",
                    "ASVspoof5.dev.enroll.txt",
                    "last_eval/ASVspoof5.eval.track_1.tsv",
                    "last_eval/ASVspoof5.eval.track_2.enroll.tsv",
                )
            ],
        ]
        write_json(output / "progress.json", {"status": "running", "stage": "hashing_inputs"})
        identity = {"config": cfg, "sha256": {str(path): digest(path) for path in inputs}}
        path = output / "provenance.json"
        if path.exists() and json.loads(path.read_text()) != identity:
            message = "SASV experiment identity changed; use a new output directory"
            raise ValueError(message)
        write_json(path, identity)
        manifest_path = output / "manifest.json"
        if not manifest_path.exists():
            write_json(output / "progress.json", {"status": "running", "stage": "preparing_native_pairs"})
            write_json(manifest_path, prepare(cfg))
        manifest = json.loads(manifest_path.read_text())
        LOGGER.info(
            "Native pairs ready: support=%d tune=%d held-out pilot=%d",
            len(manifest["support"]),
            len(manifest["tune"]),
            len(manifest["pilot_held_out"]),
        )
        model = SASVALM(cfg)
        write_json(output / "load.json", model.audit)
        smoke(model, manifest["dev_baseline"][:2], output)
        # Small engineering/development baseline before fitting any direction.
        evaluate_group(model, manifest["dev_guard"], "dev_guard_baseline", {"baseline": None}, {}, output)
        audio = {}
        for name in ("support", "known", "tune", "dev_guard"):
            path = output / f"audio_{name}.pt"
            if not path.exists():
                save_tensor(path, encode(model, manifest[name]))
            audio[name] = torch.load(path, weights_only=True, map_location="cpu")
        path = output / "directions.pt"
        if not path.exists():
            started = time.perf_counter()
            directions = fit_directions(model, audio, manifest, cfg["layers"])
            save_tensor(path, directions)
            write_json(
                output / "direction_fit.json",
                {"wall_seconds": time.perf_counter() - started, "names": list(directions), "training": False},
            )
        directions = torch.load(path, weights_only=True, map_location="cpu")
        # Every family includes the same frozen fallback; no evaluation labels enter selection.
        candidates = candidate_settings(cfg)
        tuning = evaluate_group(model, manifest["tune"], "tune", candidates, directions, output, audio["tune"])
        tuning_metrics = {key: report(manifest["tune"], values)["pooled"] for key, values in tuning.items()}
        best = {}
        for family in FAMILIES:
            options = [key for key, setting in candidates.items() if setting is None or setting["family"] == family]
            options = [
                key
                for key in options
                if tuning_metrics[key]["binary"]["bonafide_recall"]
                >= tuning_metrics["baseline"]["binary"]["bonafide_recall"] - 0.05
            ]
            best[family] = min(options, key=lambda key: rank(tuning_metrics[key], candidates[key]))
        guards = {key: candidates[key] for key in {"baseline", *best.values()}}
        dev = evaluate_group(model, manifest["dev_guard"], "dev_guard", guards, directions, output, audio["dev_guard"])
        dev_metrics = {key: report(manifest["dev_guard"], values)["pooled"] for key, values in dev.items()}
        selected = {
            family: key if allowed(dev_metrics[key], dev_metrics["baseline"]) else "baseline"
            for family, key in best.items()
        }
        settings = {key: candidates[key] for key in {"baseline", *selected.values()}}
        nonzero = [key for key in selected.values() if key != "baseline"]
        if nonzero:
            reference = min(nonzero, key=lambda key: rank(tuning_metrics[key], candidates[key]))
            setting = candidates[reference]
            settings["random_control"] = {**setting, "direction": f"random_{setting['layer']}"}
            settings["reverse_control"] = {**setting, "strength": -setting["strength"]}
        write_json(
            output / "selection.json",
            {
                "selected": selected,
                "tune_best_before_dev_guard": best,
                "evaluated_settings": settings,
                "criterion": "Tuning binary F1, native macro F1, accuracy; dev/native retention guard; zero fallback",
                "test_labels_used": False,
            },
        )
        evaluate_group(model, manifest["dev_baseline"], "configured_dev_2048", {"baseline": None}, {}, output)
        predictions = evaluate_group(model, manifest["pilot_full"], "native_eval_20k", settings, directions, output)
        indices = [i for i, row in enumerate(manifest["pilot_full"]) if row["pool"] == "test"]
        held_out = {
            key: report(manifest["pilot_held_out"], [values[i] for i in indices]) for key, values in predictions.items()
        }
        write_json(
            output / "report.json",
            {
                "status": "pilot_complete",
                "selected": selected,
                "speaker_held_out": held_out,
                "native_20k": {key: report(manifest["pilot_full"], values) for key, values in predictions.items()},
                "full_native_pair_count": manifest["full_native_pair_count"],
                "full_native_evaluation_complete": False,
                "note": "20K native-pair pilot, not the full 834,536-pair manifest or the single-audio benchmark.",
            },
        )
        write_json(output / "progress.json", {"status": "pilot_complete", "full_native_evaluation_complete": False})


def main() -> None:
    """Read the fixed native SASV pilot configuration."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/alm_sasv/default.yaml"))
    args = parser.parse_args()
    run(yaml.safe_load(args.config.read_text()))


if __name__ == "__main__":
    main()
