"""Fit frozen-model steering directions from one ElevenLabs voice."""

import json
import logging
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_file, save_file
from torch import Tensor

from src.speaker_steering.protocol import make_plan, order_key, records
from src.steering.external_eval import digest, provenance, validate_identity
from src.steering.external_protocol import ExternalAudio, inventory, read_elevenlabs
from src.steering.full_eval import Scorer, build_scorers, smoke_check
from src.steering.inference import load_audio, run_model
from src.steering.midsteer import RankOneSteering, fit_transport
from src.steering.protocol import Trial
from src.steering.run import admissible, candidate_metrics, feature_direction, write_json
from src.steering.run_midsteer import sampled_tokens
from src.steering.sae import TopKSAE

log = logging.getLogger(__name__)
FAMILIES = ("speaker_dense", "speaker_sae", "speaker_affine", "speaker_gated")


def load_sae(cfg: DictConfig, layer: int) -> TopKSAE:
    """Restore the existing training-only dictionary, without SAE retraining.

    Args:
        cfg: Reference run, dictionary size and device.
        layer: One-based encoder layer.

    Returns:
        Frozen existing dictionary.
    """
    weights = load_file(Path(cfg.steering.reference_run) / f"sae_layer_{layer}.safetensors")
    sae = TopKSAE(len(weights["center"]), int(cfg.steering.expansion), int(cfg.steering.k))
    sae.load_state_dict(weights, strict=True)
    return sae.to(str(cfg.steering.device)).requires_grad_(False).eval()


def additive(hidden: Tensor, vector: Tensor, strength: float) -> Tensor:
    """Apply a fixed spoof-minus-bonafide contrast to every input token.

    Args:
        hidden: Input activations.
        vector: Fitted contrast in original activation units.
        strength: Selected signed strength.

    Returns:
        Corrected activations, with an exact zero-strength identity.
    """
    return hidden if strength == 0 else hidden + strength * vector.to(hidden)


def transform(
    cfg: DictConfig,
    root: Path,
    method: str,
    layer: int,
    strength: float,
) -> Any:
    """Restore a label-independent intervention from fitted artifacts.

    Args:
        cfg: Device and dictionary references.
        root: Adaptation artifact directory.
        method: Dense, SAE, transport or prespecified SAE control.
        layer: Selected layer.
        strength: Selected strength.

    Returns:
        Callable activation transform, or None for baseline fallback.
    """
    if strength == 0:
        return None
    state = load_file(root / f"directions_layer_{layer}.safetensors", device=str(cfg.steering.device))
    if method == "speaker_dense":
        return partial(additive, vector=state["dense"], strength=strength)
    if method.startswith("speaker_sae"):
        sae = load_sae(cfg, layer)
        signs = state["sae_signs"]
        if method.endswith("random"):
            order = torch.randperm(len(signs), generator=torch.Generator().manual_seed(int(cfg.seed))).to(signs.device)
            signs = signs[order]
        if method.endswith("reversed"):
            strength = -strength
        return partial(sae.steer, direction=signs, strength=strength)
    prefix = "gated" if method == "speaker_gated" else "affine"
    adapter = RankOneSteering(state[f"{prefix}_anchor"], state[f"{prefix}_read"], state[f"{prefix}_write"])
    return partial(adapter.steer, strength=strength, gated=prefix == "gated")


def select_candidates(
    cfg: DictConfig,
    output: Path,
    model: Any,
    candidates: list[dict[str, Any]],
    partitions: dict[str, list[Trial]],
    waves: dict[str, Tensor],
    threshold: float,
    baseline: dict[str, float],
) -> dict[str, Any]:
    """Choose the best tuning candidate that also passes development retention.

    Args:
        cfg: Fixed selection and inference settings.
        output: Fitted direction directory.
        model: Frozen detector.
        candidates: Tuning results; development metrics are populated in place.
        partitions: Support, tuning and development metadata.
        waves: Corresponding decoded waveforms, without test audio.
        threshold: Original fixed development threshold.
        baseline: Original detector selection metrics.

    Returns:
        One selected candidate per family, including exact zero fallback.
    """
    settings = cfg.steering
    selected: dict[str, Any] = {}
    for method in FAMILIES:
        ordered = sorted(
            (
                row
                for row in candidates
                if row["method"] == method
                and row["tune_bonafide_rejection"]
                <= baseline["tune_bonafide_rejection"] + settings.max_bonafide_degradation
            ),
            key=lambda row: (row["tune_eer"], row["tune_min_dcf"], abs(row["strength"]), row["layer"]),
        )
        for row in ordered:
            if row["strength"] != 0:
                job = Scorer(model, row["layer"], transform(cfg, output, method, row["layer"], row["strength"]))
                dev_scores = job.score(waves["dev"], int(settings.batch_size))
                row.update(
                    candidate_metrics(
                        partitions,
                        {"tune": np.asarray(row["tune_scores"]), "dev": dev_scores},
                        threshold,
                    ),
                )
                row["dev_checked"] = True
            row["admissible"] = admissible(row, baseline, settings)
            if row["admissible"]:
                selected[method] = {key: value for key, value in row.items() if key != "tune_scores"}
                break
        if method not in selected:
            message = "No admissible candidate, including zero fallback"
            raise ValueError(message)
        log.info("Selected %s: %s", method, selected[method])
    return selected


def select_nonzero_diagnostics(
    cfg: DictConfig,
    output: Path,
    model: Any,
    candidates: list[dict[str, Any]],
    partitions: dict[str, list[Trial]],
    waves: dict[str, Tensor],
    threshold: float,
    baseline: dict[str, float],
) -> dict[str, Any]:
    """Retain best nonzero tuning candidates as diagnostics, even if inadmissible.

    Args:
        cfg: Fixed inference settings and retention limits.
        output: Fitted direction directory.
        model: Frozen detector.
        candidates: Candidate tuning metrics, without evaluation observations.
        partitions: Fitting, tuning and development metadata.
        waves: Decoded selection waveforms only.
        threshold: Original fixed development threshold.
        baseline: Original detector retention metrics.

    Returns:
        Explicitly diagnostic candidates; these do not replace guarded selection.
    """
    selected: dict[str, Any] = {}
    for method in FAMILIES:
        row = min(
            (row for row in candidates if row["method"] == method and row["strength"] != 0),
            key=lambda row: (row["tune_eer"], row["tune_min_dcf"], abs(row["strength"]), row["layer"]),
        )
        job = Scorer(model, row["layer"], transform(cfg, output, method, row["layer"], row["strength"]))
        scores = {"tune": np.asarray(row["tune_scores"]), "dev": job.score(waves["dev"], int(cfg.steering.batch_size))}
        metric = candidate_metrics(partitions, scores, threshold)
        result = {key: value for key, value in row.items() if key != "tune_scores"}
        result.update(metric)
        result.update(
            {"dev_checked": True, "admissible": admissible(metric, baseline, cfg.steering), "diagnostic_only": True},
        )
        selected[method] = result
        log.info("Nonzero diagnostic %s: %s", method, result)
    return selected


def fit(cfg: DictConfig, output: Path) -> None:
    """Fit bounded support directions and freeze selection before test scoring.

    Args:
        cfg: Speaker, split budgets, fixed candidate grid and retention rules.
        output: Exclusively locked adaptation directory.
    """
    settings = cfg.steering
    manifest = inventory(read_elevenlabs(Path(settings.external_root)))
    prior_manifest = json.loads((Path(settings.source_external_run) / "manifest.json").read_text())
    if manifest != prior_manifest:
        message = "ElevenLabs inventory changed since the baseline evaluation"
        raise ValueError(message)
    rows = records(manifest["recordings"])
    plan = make_plan(
        rows,
        str(settings.speaker),
        int(cfg.seed),
        int(settings.support_spoof),
        int(settings.tune_spoof),
        int(settings.bonafide_per_split),
    )
    identity = provenance(cfg, manifest)
    identity["reference_adaptation"] = identity.pop("adaptation")
    identity["adaptation"] = (
        f"Speaker {settings.speaker}: fit directions on prepared support audio; select on separate sources. "
        "Frozen detector and existing training-only SAE dictionaries; no evaluation-dependent selection."
    )
    identity["speaker_plan"] = plan
    identity["speaker_sources"] = {
        str(path): digest(path) for path in sorted(Path("src/speaker_steering").glob("*.py"))
    }
    identity["speaker_config_sha256"] = digest(Path("configs/steering/elevenlabs_speaker.yaml"))
    identity["fit_settings"] = {
        key: OmegaConf.to_container(settings[key]) if OmegaConf.is_config(settings[key]) else settings[key]
        for key in (
            "layers",
            "strengths",
            "known_per_class",
            "selected_features",
            "tokens_per_clip",
            "ridge",
            "max_bonafide_degradation",
            "max_dev_eer_degradation",
        )
    }
    validate_identity(output, identity)
    write_json(output / "provenance.json", {"identity": identity, "config": OmegaConf.to_container(cfg, resolve=True)})
    write_json(output / "plan.json", plan)
    if settings.prepare_only:
        log.info("Prepared speaker %s: %s", settings.speaker, plan["counts"])
        return
    if (output / "selection.json").exists():
        log.info("Existing immutable selection retained; start evaluation separately")
        return
    jobs, aliases = build_scorers(cfg)
    smoke = smoke_check(jobs, aliases, cfg)
    model = jobs["baseline"].model
    del jobs, aliases
    model.requires_grad_(False).eval()
    original = json.loads((Path(settings.reference_run) / "manifest.json").read_text())
    partitions = {key: [row.trial() for row in records(plan[key])] for key in ("support", "tune")}
    partitions["dev"] = [Trial(**row) for row in original["dev"]]
    known = []
    for label in (0, 1):
        candidates = sorted(
            (row for row in original["sae_train"] if row["label"] == label),
            key=lambda row: order_key(row["utterance"], int(cfg.seed)),
        )
        if len(candidates) < int(settings.known_per_class):
            message = "Insufficient training-only transport reference"
            raise ValueError(message)
        known.extend(Trial(**row) for row in candidates[: int(settings.known_per_class)])
    write_json(output / "reference_manifest.json", {"known": [row.__dict__ for row in known], "dev": original["dev"]})
    waves = {
        key: torch.stack([wave for _, wave in ExternalAudio(records(plan[key]), int(cfg.data.num_samples))])
        for key in ("support", "tune")
    }
    waves["dev"] = load_audio(partitions["dev"], int(cfg.data.num_samples))
    waves["known"] = load_audio(known, int(cfg.data.num_samples))
    threshold = float(json.loads((Path(settings.reference_run) / "selection.json").read_text())["threshold"])
    baseline_job = Scorer(model, int(settings.layers[0]))
    base_scores = {key: baseline_job.score(waves[key], int(settings.batch_size)) for key in ("tune", "dev")}
    baseline = candidate_metrics(partitions, base_scores, threshold)
    labels = np.array([row.label for row in partitions["support"]])
    known_labels = np.array([row.label for row in known])
    candidates = []
    for layer in map(int, settings.layers):
        block = model.frontend.model.encoder.layers[layer - 1]
        captured = {
            key: run_model(model, waves[key], block, int(settings.batch_size), capture=True)[1]
            for key in ("support", "known")
        }
        support_hidden, known_hidden = captured["support"], captured["known"]
        assert support_hidden is not None and known_hidden is not None
        sae = load_sae(cfg, layer)
        signs = feature_direction(sae, support_hidden, labels, int(settings.selected_features))
        means = support_hidden.mean(dim=1)
        dense = means[labels == 1].mean(0) - means[labels == 0].mean(0)
        support_tokens = sampled_tokens(support_hidden, int(settings.tokens_per_clip))
        known_tokens = sampled_tokens(known_hidden, int(settings.tokens_per_clip))
        groups = (
            torch.cat((support_tokens[labels == 0], known_tokens[known_labels == 0]))
            .flatten(0, 1)
            .to(str(settings.device)),
            known_tokens[known_labels == 1].flatten(0, 1).to(str(settings.device)),
            support_tokens[labels == 1].flatten(0, 1).to(str(settings.device)),
        )
        state = {"dense": dense.cpu().contiguous(), "sae_signs": signs.cpu().contiguous()}
        for prefix in ("affine", "gated"):
            adapter = fit_transport(groups, float(settings.ridge), anchored=prefix == "gated")
            state.update({f"{prefix}_{key}": value.cpu().contiguous() for key, value in adapter.state_dict().items()})
        save_file(state, output / f"directions_layer_{layer}.safetensors")
        zero = sae.steer(support_hidden[:1].to(str(settings.device)), signs, 0)
        torch.testing.assert_close(zero.cpu(), support_hidden[:1], rtol=0, atol=0)
        log.info("Fitted layer %d; frozen SAE reused, no detector updates", layer)
        del captured, support_hidden, known_hidden, support_tokens, known_tokens, groups, sae, adapter
        for method in FAMILIES:
            for strength in sorted(set(map(float, settings.strengths)) | {0.0}):
                job = Scorer(model, layer, transform(cfg, output, method, layer, strength))
                tune_scores = (
                    base_scores["tune"] if strength == 0 else job.score(waves["tune"], int(settings.batch_size))
                )
                metrics = candidate_metrics(partitions, {"tune": tune_scores, "dev": base_scores["dev"]}, threshold)
                candidates.append(
                    {
                        "method": method,
                        "layer": layer,
                        "strength": strength,
                        **metrics,
                        "tune_scores": tune_scores.tolist(),
                        "dev_checked": strength == 0,
                    },
                )
    selected = select_candidates(cfg, output, model, candidates, partitions, waves, threshold, baseline)
    nonzero = select_nonzero_diagnostics(cfg, output, model, candidates, partitions, waves, threshold, baseline)
    write_json(
        output / "selection.json",
        {
            "threshold": threshold,
            "baseline": baseline,
            "selected": selected,
            "nonzero_diagnostics": nonzero,
            "candidates": candidates,
            "test_scores_used": False,
            "criterion": "OY tuning EER/minimum cost; genuine retention; ASVspoof development retention; zero fallback",
        },
    )
    write_json(output / "baseline_selection_scores.json", {key: value.tolist() for key, value in base_scores.items()})
    smoke.update(
        {
            "status": "passed",
            "zero_sae_identity": True,
            "all_detector_parameters_frozen": all(not p.requires_grad for p in model.parameters()),
        },
    )
    write_json(output / "smoke.json", smoke)
    write_json(
        output / "progress.json",
        {"status": "adaptation_complete", "speaker": str(settings.speaker), "full_model_training": False},
    )
