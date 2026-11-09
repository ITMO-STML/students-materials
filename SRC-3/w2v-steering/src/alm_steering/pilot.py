"""Resumable, frozen-weight text-to-audio and support-derived steering pilot."""

import fcntl
import hashlib
import json
import logging
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from src.alm_steering.benchmark import benchmark
from src.alm_steering.classification import classification_rank, classification_report, retention_allowed
from src.alm_steering.directions import audio_states, corpus, text_directions
from src.alm_steering.finetune import fitting_smoke
from src.alm_steering.interventions import Intervention, steer
from src.alm_steering.run import smoke, write_json
from src.alm_steering.runtime import FrozenALM
from src.steering.midsteer import fit_transport
from src.steering.protocol import Trial, check_disjoint

log = logging.getLogger(__name__)


def cached_audio(model: FrozenALM, trials: list[Trial], path: Path, batch_size: int) -> Tensor:
    """Save projected audio in atomic chunks, preserving manifest order.

    Args:
        model: Frozen model.
        trials: Exact ordered records.
        path: Run-owned cache directory.
        batch_size: Encoding batch size.

    Returns:
        All projected tokens on CPU.
    """
    path.mkdir(exist_ok=True)
    chunks = []
    for start in range(0, len(trials), 32):
        end = min(start + 32, len(trials))
        destination = path / f"{start:06d}_{end:06d}.pt"
        if destination.exists():
            values = torch.load(destination, map_location="cpu", weights_only=True)
        else:
            values = torch.cat(
                [model.audio_embeddings(trials[i : min(i + batch_size, end)]) for i in range(start, end, batch_size)],
            )
            temporary = destination.with_suffix(".partial")
            torch.save(values, temporary)
            temporary.replace(destination)
        assert len(values) == end - start and torch.isfinite(values).all()
        chunks.append(values)
        log.info("Audio cache %s: %d/%d", path.name, end, len(trials))
    return torch.cat(chunks)


def score_group(
    model: FrozenALM,
    audio: Tensor,
    path: Path,
    batch_size: int,
    setting: dict[str, Any] | None = None,
    direction: dict[str, Tensor] | None = None,
    prompt: str | None = None,
) -> np.ndarray:
    """Cache one complete group of scores, with named settings fixed by provenance.

    Args:
        model: Frozen ALM.
        audio: Projected audio tokens.
        path: Unique score artifact for this group and setting.
        batch_size: Scoring batch size.
        setting: Layer, scope and strength, or None for baseline.
        direction: Additive or transport vectors.
        prompt: Optional prompting-only control.

    Returns:
        Bonafide-positive class log odds in manifest order.
    """
    if path.exists():
        data = json.loads(path.read_text())
        values = np.asarray(data["scores"])
        assert len(values) == len(audio) and np.isfinite(values).all()
        return values
    context = nullcontext()
    if setting and setting["strength"] != 0:
        with torch.no_grad():
            prefix, span = model.prefix(audio[:1], prompt)
        intervention = Intervention(
            strength=setting["strength"],
            scope=setting["scope"],
            prefix_length=prefix.shape[1],
            audio_span=span,
            **direction,
        )
        context = steer(model.layers[setting["layer"] - 1], intervention)
    started = time.perf_counter()
    with context:
        scores = np.concatenate(
            [model.score(audio[start : start + batch_size], prompt) for start in range(0, len(audio), batch_size)],
        )
    write_json(
        path,
        {"scores": scores.tolist(), "wall_seconds": time.perf_counter() - started, "setting": setting, "n": len(audio)},
    )
    return scores


def fit_directions(
    model: FrozenALM,
    audio: dict[str, Tensor],
    trials: dict[str, list[Trial]],
    layers: list[int],
    batch_size: int,
    output: Path,
) -> dict[str, dict[str, Tensor]]:
    """Fit text and support directions without using tuning or transfer labels.

    Args:
        model: Frozen ALM.
        audio: Training/support cached tokens.
        trials: Matching ordered records.
        layers: One-based intervention sites.
        batch_size: Inference batch size.
        output: Run directory.

    Returns:
        Layer-qualified additive and rank-one directions.
    """
    path = output / "directions.pt"
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=True)
    started = time.perf_counter()
    directions = {}
    for adapted in (False, True):
        for name, vector in text_directions(model, layers, adapted=adapted).items():
            directions[name] = {"vector": vector}
    states = {name: audio_states(model, audio[name], layers, batch_size) for name in ("support", "sae_train")}
    support_labels = torch.tensor([trial.label for trial in trials["support"]])
    train_labels = torch.tensor([trial.label for trial in trials["sae_train"]])
    for layer in layers:
        bona = states["support"][layer][support_labels == 0].flatten(0, 1)
        attack = states["support"][layer][support_labels == 1].flatten(0, 1)
        known = states["sae_train"][layer][train_labels == 1].flatten(0, 1)
        directions[f"audio_mean_{layer}"] = {"vector": attack.mean(0) - bona.mean(0)}
        transport = fit_transport(
            tuple(group.to(model.device) for group in (bona, known, attack)),
            0.01,
            anchored=False,
        )
        directions[f"audio_transport_{layer}"] = {
            "vector": transport.write.cpu(),
            "read": transport.read.cpu(),
            "anchor": transport.anchor.cpu(),
        }
    temporary = path.with_suffix(".partial")
    torch.save(directions, temporary)
    temporary.replace(path)
    write_json(
        output / "direction_fit.json",
        {"wall_seconds": time.perf_counter() - started, "names": list(directions), "text_corpus": corpus()},
    )
    return directions


def ensure_fitting_smoke(model: FrozenALM, audio: Tensor, trials: list[Trial], output: Path) -> None:
    """Run and persist the bounded fitting check once before frozen steering.

    Args:
        model: Baseline model, restored exactly after the check.
        audio: Cached support audio.
        trials: Corresponding support records.
        output: Run artifact directory.
    """
    if (output / "finetune_smoke.json").exists():
        return
    write_json(output / "progress.json", {"stage": "finetune_smoke"})
    indices = [i for label in (0, 1) for i in [j for j, trial in enumerate(trials) if trial.label == label][:4]]
    assert len(indices) == 8
    result = fitting_smoke(model, audio[indices], [trials[i].label for i in indices])
    write_json(output / "finetune_smoke.json", result)
    log.info("Fine-tuning smoke passed: %s", result)


def ensure_benchmark(
    model: FrozenALM,
    audio: Tensor,
    directions: dict[str, dict[str, Tensor]],
    layers: list[int],
    output: Path,
) -> None:
    """Persist representative compute measurements once.

    Args:
        model: Frozen model.
        audio: A fixed tuning batch.
        directions: Fitted vectors.
        layers: Prespecified layer grid.
        output: Run directory.
    """
    if (output / "compute_benchmark.json").exists():
        return
    write_json(output / "progress.json", {"stage": "compute_benchmark"})
    result = benchmark(model, audio, directions, layers[len(layers) // 2])
    write_json(output / "compute_benchmark.json", result)


def run_pilot(model: FrozenALM, cfg: DictConfig, output: Path) -> None:
    """Fit, select, and evaluate a fixed experiment matrix with resumable artifacts.

    Args:
        model: Frozen ALM.
        cfg: Hydra configuration.
        output: Run directory with a held process lock.
    """
    raw = json.loads(Path(cfg.alm.manifest).read_text())
    trials = {name: [Trial(**row) for row in rows] for name, rows in raw.items()}
    check_disjoint(trials)
    write_json(output / "manifest.json", raw)
    batch_size = int(cfg.alm.batch_size)
    layers = [int(layer) for layer in cfg.alm.layers]
    scores_dir = output / "scores"
    scores_dir.mkdir(exist_ok=True)
    audio = {}
    for name in ("support", "tune", "dev", "sae_train"):
        write_json(output / "progress.json", {"stage": "audio_cache", "group": name})
        audio[name] = cached_audio(model, trials[name], output / f"audio_{name}", batch_size)
    ensure_fitting_smoke(model, audio["support"], trials["support"], output)
    baseline = {
        name: score_group(model, audio[name], scores_dir / f"baseline_{name}.json", batch_size)
        for name in ("tune", "dev")
    }
    threshold = 0.0  # Fixed class-answer argmax; never optimized on evaluation labels.
    metrics = {name: classification_report(trials[name], baseline[name])["pooled"] for name in baseline}
    write_json(output / "baseline_validation.json", {"threshold": threshold, "metrics": metrics})
    write_json(output / "progress.json", {"stage": "direction_fit"})
    directions = fit_directions(model, audio, trials, layers, batch_size, output)
    ensure_benchmark(model, audio["tune"][:batch_size], directions, layers, output)
    families = [
        "text_mean_base",
        "text_probe_base",
        "text_mean_adapted",
        "text_probe_adapted",
        "audio_mean",
        "audio_transport",
    ]
    selected = {}
    candidates = []
    for family in families:
        scopes = list(cfg.alm.scopes) if family.startswith("text") else ["audio"]
        for scope in scopes:
            method = f"{family}__{scope}"
            best = {
                "family": family,
                "scope": scope,
                "layer": layers[0],
                "strength": 0.0,
                "tune": metrics["tune"],
                "dev": metrics["dev"],
                "admissible": True,
            }
            for layer in layers:
                for strength in (float(value) for value in cfg.alm.strengths if value != 0):
                    setting = {"family": family, "scope": scope, "layer": layer, "strength": strength}
                    slug = f"{method}_L{layer}_a{strength}"
                    write_json(
                        output / "progress.json",
                        {"stage": "selection", "candidate": slug, "completed": len(candidates)},
                    )
                    tune_scores = score_group(
                        model,
                        audio["tune"],
                        scores_dir / f"{slug}_tune.json",
                        batch_size,
                        setting,
                        directions[f"{family}_{layer}"],
                    )
                    tune = classification_report(trials["tune"], tune_scores)["pooled"]
                    admissible = tune["bonafide_recall"] >= metrics["tune"]["bonafide_recall"] - 0.05
                    dev = None
                    if admissible:
                        dev_scores = score_group(
                            model,
                            audio["dev"],
                            scores_dir / f"{slug}_dev.json",
                            batch_size,
                            setting,
                            directions[f"{family}_{layer}"],
                        )
                        dev = classification_report(trials["dev"], dev_scores)["pooled"]
                        admissible = retention_allowed(dev, metrics["dev"])
                    candidate = {**setting, "tune": tune, "dev": dev, "admissible": admissible}
                    candidates.append(candidate)
                    rank = (*classification_rank(tune), abs(strength), layer)
                    previous = (*classification_rank(best["tune"]), abs(best["strength"]), best["layer"])
                    if admissible and rank < previous:
                        best = candidate
                    write_json(output / "candidates.json", candidates)
                    log.info("Candidate %s: F1 %.4f admissible=%s", slug, tune["f1"], admissible)
            selected[method] = best
    # Freeze all choices before loading the transfer recordings.
    write_json(output / "selection.json", {"threshold": threshold, "selected": selected})
    write_json(output / "progress.json", {"stage": "pilot_test_cache"})
    audio["test"] = cached_audio(model, trials["test"], output / "audio_test", batch_size)
    reports = {}
    base_test = score_group(model, audio["test"], scores_dir / "baseline_test.json", batch_size)
    reports["baseline"] = classification_report(trials["test"], base_test)
    for method, setting in selected.items():
        write_json(output / "progress.json", {"stage": "pilot_test", "method": method})
        values = score_group(
            model,
            audio["test"],
            scores_dir / f"{method}_test.json",
            batch_size,
            setting,
            directions[f"{setting['family']}_{setting['layer']}"],
        )
        reports[method] = classification_report(trials["test"], values)
        write_json(output / "report.partial.json", reports)
    control = selected["text_mean_base__audio"].copy()
    control["strength"] = control["strength"] or 1.0
    original = directions[f"text_mean_base_{control['layer']}"]["vector"]
    random = torch.randn(original.shape, generator=torch.Generator().manual_seed(int(cfg.seed)))
    random *= original.norm() / random.norm()
    for name, vector in (("random_matched_norm", random), ("sign_reversed", -original)):
        values = score_group(
            model,
            audio["test"],
            scores_dir / f"{name}_test.json",
            batch_size,
            control,
            {"vector": vector},
        )
        reports[name] = classification_report(trials["test"], values)
    prompt = model.prompt.replace(
        "\nASSISTANT:",
        " Focus on prosody, spectral discontinuities, and synthesis artifacts.\nASSISTANT:",
    )
    values = score_group(model, audio["test"], scores_dir / "prompting_test.json", batch_size, prompt=prompt)
    reports["prompting_only"] = classification_report(trials["test"], values)
    write_json(
        output / "report.json",
        {
            "cohort": "historical speaker-disjoint 4352-recording pilot, NOT full evaluation",
            "threshold": threshold,
            "results": reports,
            "control_setting": control,
            "fine_tuning": "two-step fitting smoke only; separate approved comparator",
            "decision_rule": "spoof if logP(bonafide)-logP(spoof) < 0; positive class spoof",
        },
    )
    write_json(output / "progress.json", {"stage": "complete", "status": "complete"})


@hydra.main(version_base="1.3", config_path="../../configs", config_name="main")
def main(cfg: DictConfig) -> None:
    """Run a locked, provenance-checked ALM pilot from a detached session.

    Args:
        cfg: Main configuration with +alm=default.
    """
    torch.set_num_threads(4)
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    output = Path(cfg.alm.resume_dir or cfg.paths.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "coordinator.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        settings = OmegaConf.to_container(cfg.alm, resolve=True)
        identity_settings = {key: value for key, value in settings.items() if key != "resume_dir"}
        files = [
            *Path("src/alm_steering").glob("*.py"),
            *Path(cfg.alm.source_package).glob("*.py"),
            Path(cfg.alm.manifest),
            Path(cfg.alm.inventory),
            Path("src/steering/midsteer.py"),
            Path("src/steering/evaluation.py"),
            Path("src/steering/protocol.py"),
            Path("src/metrics/detection.py"),
        ]
        identity = {
            "settings": identity_settings,
            "seed": int(cfg.seed),
            "sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
        }
        provenance = output / "identity.json"
        if provenance.exists() and json.loads(provenance.read_text()) != identity:
            message = "Resume identity changed; use a new run directory"
            raise ValueError(message)
        write_json(provenance, identity)
        model = FrozenALM(settings)
        if not (output / "smoke.json").exists():
            smoke(model, cfg, output)
        run_pilot(model, cfg, output)


if __name__ == "__main__":
    main()
