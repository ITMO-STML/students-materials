"""Hydra entrypoint for bounded A21 support -> A22/A31 transfer experiments."""

import hashlib
import json
import logging
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import sklearn
import torch
import transformers
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_file, save_file
from sklearn.linear_model import LogisticRegression
from torch import Tensor, nn
from transformers import Wav2Vec2Config, Wav2Vec2Model, set_seed

from src.model.model import Wav2Vec2AASIST
from src.steering.evaluation import attack_report, dev_threshold
from src.steering.inference import load_audio, run_model
from src.steering.protocol import Trial, check_disjoint, read_trials, select_trials, speaker_partition
from src.steering.sae import TopKSAE, active_mean, fit_sae

log = logging.getLogger(__name__)

type FittedLayer = tuple[TopKSAE, Tensor, Tensor]


def encoder_block(model: Wav2Vec2AASIST, layer: int) -> nn.Module:
    """Resolve a one-based transformer layer with explicit bounds checking.

    Args:
        model: Baseline detector.
        layer: One-based layer index.

    Returns:
        Requested transformer block.
    """
    if not 1 <= layer <= len(model.frontend.model.encoder.layers):
        message = f"Invalid one-based transformer layer: {layer}"
        raise ValueError(message)
    return model.frontend.model.encoder.layers[layer - 1]


def write_json(path: Path, content: Any) -> None:
    """Write strict JSON for experiment provenance and results.

    Args:
        path: Destination in the new experiment directory.
        content: JSON-serializable data.
    """
    path.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")


def make_partitions(cfg: DictConfig) -> dict[str, list[Trial]]:
    """Build support/tuning/transfer subsets before running any inference.

    Args:
        cfg: Composed experiment configuration.

    Returns:
        Audited SAE-train, support, tune, dev, and test records.
    """
    root, seed, settings = Path(cfg.data.root), int(cfg.seed), cfg.steering
    evaluation = read_trials(root, "eval")
    pools = {name: [] for name in ("support", "tune", "test")}
    for trial in evaluation:
        pools[speaker_partition(trial.speaker, seed)].append(trial)
    dev = read_trials(root, "dev")
    groups = ["bonafide", *sorted({t.attack for t in dev if t.label == 1})]
    partitions = {
        "sae_train": select_trials(
            read_trials(root, "train"),
            ["bonafide", "spoof"],
            int(settings.sae_train_per_group),
            seed,
        ),
        "support": select_trials(pools["support"], ["bonafide", "A21"], int(settings.support_per_group), seed),
        "tune": select_trials(pools["tune"], ["bonafide", "A21"], int(settings.tune_per_group), seed),
        "dev": select_trials(dev, groups, int(settings.dev_per_group), seed),
        "test": select_trials(pools["test"], ["bonafide", *settings.test_attacks], int(settings.test_per_group), seed),
    }
    check_disjoint(partitions)
    return partitions


def feature_direction(sae: TopKSAE, hidden: Tensor, labels: np.ndarray, count: int) -> Tensor:
    """Fit an L2 logistic probe and select top standardized SAE coefficients.

    Args:
        sae: Fitted SAE.
        hidden: CPU support activations.
        labels: Support labels, spoof-positive.
        count: Number of selected nonconstant features.

    Returns:
        Sparse coefficient signs on the SAE device.
    """
    if count < 1:
        message = "selected_features must be positive"
        raise ValueError(message)
    device = sae.center.device
    with torch.inference_mode():
        features = torch.cat([active_mean(sae.encode(batch.to(device))).cpu() for batch in hidden.split(4)]).numpy()
    std = features.std(axis=0)
    active = std > 1e-6
    if not active.any():
        message = "No varying SAE features on support recordings"
        raise ValueError(message)
    normalized = (features[:, active] - features[:, active].mean(axis=0)) / std[active]
    probe = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=0).fit(normalized, labels)
    importance = np.zeros(features.shape[1])
    importance[active] = probe.coef_[0]
    indices = np.argsort(np.abs(importance))[-min(count, int(active.sum())) :]
    direction = np.zeros_like(importance)
    direction[indices] = np.sign(importance[indices])
    return torch.tensor(direction, dtype=torch.float32, device=device)


def candidate_metrics(
    partitions: dict[str, list[Trial]],
    scores: dict[str, np.ndarray],
    threshold: float,
) -> dict[str, float]:
    """Summarize A21 tuning quality and known-attack retention.

    Args:
        partitions: Experiment manifests.
        scores: Tune and dev score arrays.
        threshold: Frozen baseline dev threshold.

    Returns:
        Scalar selection objective and retention guardrails.
    """
    tune = attack_report(partitions["tune"], scores["tune"], threshold)["pooled"]
    dev = attack_report(partitions["dev"], scores["dev"], threshold)["pooled"]
    return {
        "tune_eer": tune["eer"],
        "tune_min_dcf": tune["min_dcf"],
        "dev_eer": dev["eer"],
        "tune_bonafide_rejection": tune["bonafide_rejection"],
        "dev_bonafide_rejection": dev["bonafide_rejection"],
    }


def admissible(candidate: dict[str, Any], baseline: dict[str, float], settings: DictConfig) -> bool:
    """Apply retention constraints using tuning and dev observations only.

    Args:
        candidate: Candidate metrics.
        baseline: Frozen baseline metrics.
        settings: Allowed absolute degradations in fraction units.

    Returns:
        Whether the candidate satisfies all retention constraints.
    """
    return candidate["dev_eer"] <= baseline["dev_eer"] + settings.max_dev_eer_degradation and all(
        candidate[key] <= baseline[key] + settings.max_bonafide_degradation
        for key in ("tune_bonafide_rejection", "dev_bonafide_rejection")
    )


def apply_intervention(hidden: Tensor, *, method: str, strength: float, fitted: FittedLayer) -> Tensor:
    """Apply a label-independent intervention using already fitted parameters.

    Args:
        hidden: Original token states.
        method: baseline, reconstruction, dense, or sae.
        strength: Intervention strength.
        fitted: SAE, feature signs, and dense direction.

    Returns:
        Modified token states.
    """
    sae, signs, dense = fitted
    if method == "baseline":
        return hidden
    if method == "reconstruction":
        return sae(hidden).to(hidden.dtype)
    if method == "dense":
        return hidden + strength * dense
    if method == "sae":
        return sae.steer(hidden, signs, strength)
    message = f"Unknown intervention: {method}"
    raise ValueError(message)


def tune_layer(
    model: Wav2Vec2AASIST,
    waves: dict[str, Tensor],
    partitions: dict[str, list[Trial]],
    layer: int,
    fitted: FittedLayer,
    baseline: dict[str, float],
    threshold: float,
    settings: DictConfig,
) -> list[dict[str, Any]]:
    """Compare strength candidates without accessing transfer recordings.

    Args:
        model: Frozen detector.
        waves: Non-test waveforms.
        partitions: Fixed manifests.
        layer: One-based intervention layer.
        fitted: Fitted dictionary and directions.
        baseline: Unmodified selection metrics.
        threshold: Frozen dev operating threshold.
        settings: Search and retention settings.

    Returns:
        Candidate metrics, including the zero-strength fallback.
    """
    candidates = []
    for method in ("dense", "sae"):
        for strength in sorted({0.0, *map(float, settings.strengths)}):
            if strength == 0:
                metrics = dict(baseline)
            else:
                transform = partial(apply_intervention, method=method, strength=strength, fitted=fitted)
                block = model.frontend.model.encoder.layers[layer - 1]
                scored = {
                    name: run_model(model, waves[name], block, int(settings.batch_size), transform)[0]
                    for name in ("tune", "dev")
                }
                metrics = candidate_metrics(partitions, scored, threshold)
            candidate = {"method": method, "layer": layer, "strength": strength, **metrics}
            candidate["admissible"] = admissible(candidate, baseline, settings)
            candidates.append(candidate)
            log.info("Candidate: %s", candidate)
    return candidates


@hydra.main(version_base="1.3", config_path="../../configs", config_name="main")
def main(cfg: DictConfig) -> None:
    """Fit bounded SAE/dense interventions and report held-out per-attack changes.

    Args:
        cfg: Hydra config; smoke budgets are defined in steering/a21.yaml.
    """
    set_seed(int(cfg.seed))
    torch.set_num_threads(4)
    settings = cfg.steering
    output = Path(cfg.paths.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    partitions = make_partitions(cfg)
    write_json(output / "manifest.json", {name: [asdict(t) for t in trials] for name, trials in partitions.items()})
    checkpoint = Path(settings.checkpoint).resolve()
    saved = OmegaConf.load(checkpoint.parent / ".hydra/config.yaml")
    if int(saved.data.num_samples) != int(cfg.data.num_samples):
        message = "Experiment waveform length differs from baseline training configuration"
        raise ValueError(message)
    encoder_cfg = Wav2Vec2Config.from_pretrained(str(saved.model.wav2vec2.model_name), local_files_only=True)
    model = Wav2Vec2AASIST(
        saved.model.wav2vec2,
        saved.model.aasist,
        hydra.utils.instantiate(saved.model.criterion),
        Wav2Vec2Model(encoder_cfg),
    )
    model.load_state_dict(load_file(checkpoint / "model.safetensors"), strict=True)
    model.eval().requires_grad_(False).to(str(settings.device))
    with (checkpoint / "model.safetensors").open("rb") as handle:
        checkpoint_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    write_json(
        output / "provenance.json",
        {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_hash,
            "protocol": "few-shot eval adaptation; speaker/source-disjoint; A22/A31 never used for selection",
            "score": "softmax(logits)[0], higher is bonafide",
            "layer_indexing": "one-based block output before final norm",
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "sklearn": sklearn.__version__,
            "source_sha256": {
                str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(
                    [
                        *Path("src/steering").glob("*.py"),
                        Path("src/model/model.py"),
                        Path("src/data/dataset.py"),
                        Path("src/metrics/detection.py"),
                    ],
                )
            },
            "seed": int(cfg.seed),
            "smoke_only": settings.stage == "smoke",
            "stage": str(settings.stage),
            "sae": "per-token TopK, not Whisper paper BatchTopK; residual-preserving multiplicative steering",
        },
    )
    waves = {
        name: load_audio(trials, int(cfg.data.num_samples)) for name, trials in partitions.items() if name != "test"
    }
    batch_size = int(settings.batch_size)
    base_scores = {}
    fitted = {}
    candidates = []
    baseline: dict[str, float] = {}
    threshold = None
    for layer_value in settings.layers:
        layer = int(layer_value)
        block = encoder_block(model, layer)
        cached = {}
        for name, audio in waves.items():
            score, hidden = run_model(model, audio, block, batch_size, capture=name in {"sae_train", "support", "dev"})
            base_scores[name] = score
            if hidden is not None:
                cached[name] = hidden
        if threshold is None:
            threshold = dev_threshold(partitions["dev"], base_scores["dev"])
            baseline = candidate_metrics(partitions, base_scores, threshold)
        train_hidden = cached["sae_train"]
        token_indices = torch.linspace(
            0,
            train_hidden.shape[1] - 1,
            min(int(settings.tokens_per_clip), train_hidden.shape[1]),
        ).long()
        tokens = train_hidden[:, token_indices].reshape(-1, train_hidden.shape[-1])
        sae = TopKSAE(tokens.shape[-1], int(settings.expansion), int(settings.k)).to(str(settings.device))
        log.info("Fitting layer %s SAE: %s training tokens, %s steps", layer, len(tokens), settings.sae_steps)
        losses = fit_sae(sae, tokens, int(settings.sae_steps), int(settings.sae_batch_size), int(cfg.seed))
        with torch.inference_mode():
            validation = cached["dev"][:, token_indices].reshape(-1, tokens.shape[-1]).to(str(settings.device))
            reconstruction_nmse = float(
                (sae(validation) - validation).square().mean()
                / (validation - sae.center).square().mean().clamp_min(1e-8),
            )
        labels = np.array([t.label for t in partitions["support"]])
        direction = feature_direction(sae, cached["support"], labels, int(settings.selected_features))
        means = cached["support"].mean(dim=1)
        dense = (means[labels == 1].mean(dim=0) - means[labels == 0].mean(dim=0)).to(str(settings.device))
        save_file(
            {k: v.contiguous().cpu() for k, v in sae.state_dict().items()},
            output / f"sae_layer_{layer}.safetensors",
        )
        save_file(
            {"sae_direction": direction.cpu(), "dense_direction": dense.cpu()},
            output / f"directions_layer_{layer}.safetensors",
        )
        write_json(
            output / f"sae_layer_{layer}_fit.json",
            {"losses": losses, "dev_reconstruction_nmse": reconstruction_nmse, "training_tokens": len(tokens)},
        )
        fitted[layer] = (sae, direction, dense)
        candidates.extend(tune_layer(model, waves, partitions, layer, fitted[layer], baseline, threshold, settings))
        del cached
    if threshold is None:
        message = "At least one transformer layer is required"
        raise ValueError(message)
    selected = {}
    for method in ("dense", "sae"):
        eligible = [c for c in candidates if c["method"] == method and c["admissible"]]
        selected[method] = min(
            eligible,
            key=lambda c: (c["tune_eer"], c["tune_min_dcf"], abs(c["strength"]), c["layer"]),
        )
    # Commit selection to disk before the held-out audio is loaded or scored.
    write_json(
        output / "selection.json",
        {"threshold": threshold, "baseline": baseline, "candidates": candidates, "selected": selected},
    )
    write_json(
        output / "baseline_calibration_scores.json",
        {
            name: {"utterances": [t.utterance for t in partitions[name]], "scores": base_scores[name].tolist()}
            for name in ("tune", "dev")
        },
    )
    log.info("Selection fixed; now scoring held-out attacks: %s", settings.test_attacks)
    test_waves = load_audio(partitions["test"], int(cfg.data.num_samples))
    sae_layer = int(selected["sae"]["layer"])
    sae, direction, _ = fitted[sae_layer]
    random_order = torch.randperm(direction.numel(), generator=torch.Generator().manual_seed(int(cfg.seed)))
    random_direction = direction[random_order.to(direction.device)]
    reports = {}
    scores_to_save = {}
    for method in ("baseline", "dense", "sae", "reconstruction", "random_sae"):
        layer = int(selected["dense"]["layer"]) if method == "dense" else sae_layer
        dictionary, signs, dense = fitted[layer]
        strength = float(selected[method]["strength"]) if method in selected else float(selected["sae"]["strength"])

        final_transform = partial(
            apply_intervention,
            method="sae" if method == "random_sae" else method,
            strength=strength,
            fitted=(dictionary, random_direction if method == "random_sae" else signs, dense),
        )

        scores, _ = run_model(
            model,
            test_waves,
            model.frontend.model.encoder.layers[layer - 1],
            batch_size,
            final_transform,
        )
        scores_to_save[method] = scores.tolist()
        reports[method] = attack_report(partitions["test"], scores, threshold)
        for attack, values in reports[method].items():
            reference = reports["baseline"][attack]
            values["delta_eer_pp"] = 100 * (values["eer"] - reference["eer"])
            values["delta_min_dcf"] = values["min_dcf"] - reference["min_dcf"]
            values["delta_fixed_dcf"] = values["fixed_dcf"] - reference["fixed_dcf"]
        log.info(
            "%s: %s",
            method,
            {attack: reports[method][attack] for attack in ("A21", "A22", "A31") if attack in reports[method]},
        )
    write_json(
        output / "scores.json",
        {"utterances": [t.utterance for t in partitions["test"]], "scores": scores_to_save},
    )
    write_json(
        output / "report.json",
        {
            "threshold": threshold,
            "results": reports,
            "interpretation": (
                "Smoke run only; small cohorts do not establish adaptation efficacy."
                if settings.stage == "smoke"
                else "Single-seed exploratory pilot; cohorts extend the smoke subsets; uncertainty is not estimated."
            ),
        },
    )
    log.info("Finished bounded experiment: %s", output)


if __name__ == "__main__":
    main()
