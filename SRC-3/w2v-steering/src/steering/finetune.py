"""Supervised A21 adaptation on the exact SAE pilot cohorts."""

import csv
import hashlib
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_file, save_file
from torch import Tensor, nn
from transformers import Wav2Vec2Config, Wav2Vec2Model, set_seed

from src.model.model import Wav2Vec2AASIST
from src.steering.evaluation import attack_report
from src.steering.inference import load_audio, run_model
from src.steering.protocol import Trial, check_disjoint
from src.steering.run import admissible, candidate_metrics, write_json

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

log = logging.getLogger(__name__)


def train_epoch(
    model: nn.Module,
    waves: Tensor,
    labels: Tensor,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    accumulation: int,
    seed: int,
    *,
    bf16: bool,
    max_updates: int | None = None,
) -> dict[str, float]:
    """Train with shuffled support examples and exact gradient accumulation.

    Args:
        model: Trainable detector.
        waves: CPU support waveforms.
        labels: CPU support labels.
        optimizer: Optimizer owning only detector parameters.
        batch_size: Microbatch size.
        accumulation: Microbatches per optimizer update.
        seed: Shuffle seed.
        bf16: Enable CUDA bfloat16 autocast.
        max_updates: Optional bounded smoke budget.

    Returns:
        Mean training loss, number of examples, and optimizer update count.
    """
    if batch_size < 1 or accumulation < 1 or len(waves) != len(labels) or len(waves) == 0:
        message = "Require aligned nonempty inputs and positive batch/accumulation sizes"
        raise ValueError(message)
    if max_updates is not None and max_updates < 1:
        message = "max_updates must be positive"
        raise ValueError(message)
    device = next(model.parameters()).device
    model.train()
    order = torch.randperm(len(waves), generator=torch.Generator().manual_seed(seed))
    chunks = list(order.split(batch_size * accumulation))
    if max_updates is not None:
        chunks = chunks[:max_updates]
    loss_sum = 0.0
    examples = 0
    for group in chunks:
        optimizer.zero_grad(set_to_none=True)
        for indices in group.split(batch_size):
            inputs, targets = waves[indices].to(device), labels[indices].to(device)
            # PyTorch 2.7 annotates autocast.__exit__ more broadly than the context-manager protocol.
            autocast = cast(
                "AbstractContextManager[None]",
                torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16 and device.type == "cuda"),
            )
            with autocast:
                loss = model(
                    input_values=inputs,
                    attention_mask=torch.ones_like(inputs, dtype=torch.long),
                    labels=targets,
                )["loss"]
            if not torch.isfinite(loss):
                message = "Nonfinite supervised adaptation loss"
                raise ValueError(message)
            (loss * (len(indices) / len(group))).backward()
            loss_sum += float(loss.detach()) * len(indices)
            examples += len(indices)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
    model.eval()
    return {"loss": loss_sum / examples, "examples": float(examples), "updates": float(len(chunks))}


def build_model(cfg: DictConfig) -> Wav2Vec2AASIST:
    """Load the existing detector with all weights trainable and no downloads.

    Args:
        cfg: Supervised adaptation configuration.

    Returns:
        Detector in evaluation mode on the configured device.
    """
    checkpoint = Path(cfg.steering.checkpoint)
    saved = OmegaConf.load(checkpoint.parent / ".hydra/config.yaml")
    if int(saved.data.num_samples) != int(cfg.data.num_samples):
        message = "Waveform length differs from the baseline"
        raise ValueError(message)
    encoder_config = Wav2Vec2Config.from_pretrained(str(saved.model.wav2vec2.model_name), local_files_only=True)
    model = Wav2Vec2AASIST(
        saved.model.wav2vec2,
        saved.model.aasist,
        hydra.utils.instantiate(saved.model.criterion),
        Wav2Vec2Model(encoder_config),
    )
    model.load_state_dict(load_file(checkpoint / "model.safetensors"), strict=True)
    model.requires_grad_(True).eval().to(str(cfg.steering.device))
    if cfg.steering.gradient_checkpointing:
        model.frontend.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def score_selection(
    model: Wav2Vec2AASIST,
    waves: dict[str, Tensor],
    partitions: dict[str, list[Trial]],
    batch_size: int,
    threshold: float,
) -> dict[str, float]:
    """Compute checkpoint selection criteria on tune/dev only.

    Args:
        model: Detector checkpoint to score.
        waves: Non-test waveforms.
        partitions: Fixed metadata manifests.
        batch_size: Inference batch size.
        threshold: Original baseline dev threshold.

    Returns:
        A21 tuning metrics and known-attack retention diagnostics.
    """
    model.eval()
    block = model.frontend.model.encoder.layers[0]
    scores = {name: run_model(model, waves[name], block, batch_size)[0] for name in ("tune", "dev")}
    return candidate_metrics(partitions, scores, threshold)


def save_weights(model: nn.Module, path: Path) -> None:
    """Save a standalone adapted detector without optimizer state.

    Args:
        model: Trained detector.
        path: New experiment artifact destination.
    """
    save_file({key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}, path)


def fit_comparator(
    model: Wav2Vec2AASIST,
    waves: dict[str, Tensor],
    partitions: dict[str, list[Trial]],
    cfg: DictConfig,
    threshold: float,
    output: Path,
) -> dict[str, Any]:
    """Smoke-test, reset, then fit and select supervised checkpoints.

    Args:
        model: Baseline detector with trainable parameters.
        waves: Support/tune/dev audio.
        partitions: Exact reference manifests.
        cfg: Training settings.
        threshold: Fixed threshold from the reference run.
        output: Artifact directory.

    Returns:
        Selection metadata saved before final evaluation.
    """
    settings = cfg.steering
    labels = torch.tensor([t.label for t in partitions["support"]])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings.learning_rate),
        weight_decay=float(settings.weight_decay),
    )
    before = score_selection(model, waves, partitions, int(settings.batch_size), threshold)
    smoke = train_epoch(
        model,
        waves["support"],
        labels,
        optimizer,
        int(settings.train_batch_size),
        int(settings.accumulation_steps),
        int(cfg.seed),
        bf16=bool(settings.bf16),
        max_updates=int(settings.smoke_steps),
    )
    write_json(output / "smoke.json", smoke)
    log.info("Supervised fitting smoke completed: %s; resetting weights and optimizer", smoke)
    model.load_state_dict(load_file(Path(settings.checkpoint) / "model.safetensors"), strict=True)
    del optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings.learning_rate),
        weight_decay=float(settings.weight_decay),
    )
    set_seed(int(cfg.seed))
    history = [{"epoch": 0, "admissible": True, **before}]
    best_key = (float("inf"), float("inf"))
    guarded_key = (before["tune_eer"], before["tune_min_dcf"])
    best_epoch, guarded_epoch = 0, 0
    for epoch in range(1, int(settings.epochs) + 1):
        trained = train_epoch(
            model,
            waves["support"],
            labels,
            optimizer,
            int(settings.train_batch_size),
            int(settings.accumulation_steps),
            int(cfg.seed) + epoch,
            bf16=bool(settings.bf16),
        )
        metrics = score_selection(model, waves, partitions, int(settings.batch_size), threshold)
        allowed = admissible(metrics, before, settings)
        history.append({"epoch": epoch, **trained, **metrics, "admissible": allowed})
        key = (metrics["tune_eer"], metrics["tune_min_dcf"])
        if key < best_key:
            best_key, best_epoch = key, epoch
            save_weights(model, output / "best_a21.safetensors")
        if allowed and key < guarded_key:
            guarded_key, guarded_epoch = key, epoch
            save_weights(model, output / "guarded.safetensors")
        write_json(output / "training_history.json", history)
        log.info("Epoch result: %s", history[-1])
    result = {
        "best_a21_epoch": best_epoch,
        "guarded_epoch": guarded_epoch,
        "history": history,
        "threshold": threshold,
        "protocol": "best_a21 excludes epoch zero; guarded permits baseline fallback",
    }
    write_json(output / "selection.json", result)
    return result


def evaluate_comparator(
    model: Wav2Vec2AASIST,
    partitions: dict[str, list[Trial]],
    selected: dict[str, Any],
    cfg: DictConfig,
    output: Path,
) -> None:
    """Add selected supervised checkpoints to the prior per-attack comparison.

    Args:
        model: Detector whose weights will be replaced by selected checkpoints.
        partitions: Reference cohorts.
        selected: Fixed selection decisions.
        cfg: Experiment settings.
        output: New output directory, preserving the original SAE report.
    """
    reference = Path(cfg.steering.reference_run)
    reports = json.loads((reference / "report.json").read_text())["results"]
    saved_scores = json.loads((reference / "scores.json").read_text())
    waves = load_audio(partitions["test"], int(cfg.data.num_samples))
    for name, epoch in (("finetune_a21", selected["best_a21_epoch"]), ("finetune_guarded", selected["guarded_epoch"])):
        if epoch == 0:
            scores = np.array(saved_scores["scores"]["baseline"])
        else:
            filename = "best_a21.safetensors" if name == "finetune_a21" else "guarded.safetensors"
            model.load_state_dict(load_file(output / filename), strict=True)
            model.eval()
            scores, _ = run_model(model, waves, model.frontend.model.encoder.layers[0], int(cfg.steering.batch_size))
        saved_scores["scores"][name] = scores.tolist()
        reports[name] = attack_report(partitions["test"], scores, float(selected["threshold"]))
        for attack, values in reports[name].items():
            base = reports["baseline"][attack]
            values["delta_eer_pp"] = 100 * (values["eer"] - base["eer"])
            values["delta_min_dcf"] = values["min_dcf"] - base["min_dcf"]
            values["delta_fixed_dcf"] = values["fixed_dcf"] - base["fixed_dcf"]
        log.info("%s: %s", name, {a: reports[name][a] for a in ("pooled", "A21", "A22", "A31")})
    write_json(
        output / "report.json",
        {
            "results": reports,
            "threshold": selected["threshold"],
            "reference_run": str(reference),
            "selection": selected,
        },
    )
    write_json(output / "scores.json", saved_scores)
    with (output / "comparison.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["attack", "method", "eer_percent", "min_dcf", "fixed_dcf", "bonafide_rejection_percent"])
        for attack in reports["baseline"]:
            for method in ("baseline", "sae", "finetune_a21", "finetune_guarded"):
                row = reports[method][attack]
                writer.writerow(
                    [
                        attack,
                        method,
                        100 * row["eer"],
                        row["min_dcf"],
                        row["fixed_dcf"],
                        100 * row["bonafide_rejection"],
                    ],
                )


@hydra.main(version_base="1.3", config_path="../../configs", config_name="main")
def main(cfg: DictConfig) -> None:
    """Run the supervised comparator on a previously frozen SAE manifest.

    Args:
        cfg: Hydra configuration using steering=finetune_a21.
    """
    set_seed(int(cfg.seed))
    torch.set_num_threads(4)
    reference, output = Path(cfg.steering.reference_run), Path(cfg.paths.output_dir)
    raw = json.loads((reference / "manifest.json").read_text())
    partitions = {name: [Trial(**row) for row in rows] for name, rows in raw.items()}
    check_disjoint(partitions)
    if {t.attack for t in partitions["support"]} != {"A21", "bonafide"}:
        message = "Supervised support must contain only A21 and bonafide"
        raise ValueError(message)
    write_json(output / "manifest.json", raw)
    with (Path(cfg.steering.checkpoint) / "model.safetensors").open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    provenance = json.loads((reference / "provenance.json").read_text())
    if digest != provenance["checkpoint_sha256"]:
        message = "Supervised and SAE runs must start from identical detector weights"
        raise ValueError(message)
    write_json(
        output / "provenance.json",
        {
            "checkpoint_sha256": digest,
            "reference_run": str(reference),
            "method": "full-model supervised fine-tuning, no SAE or steering",
            "seed": int(cfg.seed),
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "selection": "A21 tuning only, with separate guarded comparator",
        },
    )
    model = build_model(cfg)
    waves = {name: load_audio(partitions[name], int(cfg.data.num_samples)) for name in ("support", "tune", "dev")}
    threshold = float(json.loads((reference / "selection.json").read_text())["threshold"])
    selected = fit_comparator(model, waves, partitions, cfg, threshold, output)
    log.info(
        "Supervised checkpoint selection fixed: best=%s, guarded=%s",
        selected["best_a21_epoch"],
        selected["guarded_epoch"],
    )
    evaluate_comparator(model, partitions, selected, cfg, output)
    log.info("Completed supervised comparison: %s", output)


if __name__ == "__main__":
    main()
