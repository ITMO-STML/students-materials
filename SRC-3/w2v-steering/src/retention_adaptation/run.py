"""Prepare or fit A21 retention experiments; evaluation is exposed via eval.py."""

import fcntl
import json
import logging
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig
from transformers import set_seed

from src.steering.evaluation import attack_report
from src.steering.finetune import build_model, save_weights
from src.steering.inference import load_audio
from src.steering.protocol import Trial
from src.steering.run import admissible, candidate_metrics, write_json

from .protocol import digest, prepare
from .training import predict_logits, train_epoch

log = logging.getLogger(__name__)


def selection_metrics(
    model: torch.nn.Module,
    waves: dict[str, torch.Tensor],
    partitions: dict[str, list[Trial]],
    batch_size: int,
    threshold: float,
) -> dict[str, Any]:
    """Measure target tuning and pooled/per-generator development retention.

    Args:
        model: Current detector.
        waves: Support, retention, tuning and development waveforms.
        partitions: Frozen cohorts.
        batch_size: Inference microbatch size.
        threshold: Frozen original baseline threshold.

    Returns:
        Existing selection metrics plus per-generator development EER.
    """
    scores = {
        name: predict_logits(model, waves[name], batch_size).softmax(-1)[:, 0].numpy() for name in ("tune", "dev")
    }
    result: dict[str, Any] = candidate_metrics(partitions, scores, threshold)
    result["dev_generator_eer"] = {
        name: row["eer"]
        for name, row in attack_report(partitions["dev"], scores["dev"], threshold).items()
        if name != "pooled"
    }
    return result


def fit(cfg: DictConfig, partitions: dict[str, list[Trial]], output: Path) -> None:
    """Train without reading test audio, selecting checkpoints on tune/dev only.

    Args:
        cfg: Experiment settings; stage is smoke or fit.
        partitions: Prepared and audited cohorts.
        output: Exclusively owned artifact directory.
    """
    settings = cfg.steering
    model = build_model(cfg)
    waves = {
        name: load_audio(partitions[name], int(cfg.data.num_samples)) for name in ("support", "retain", "tune", "dev")
    }
    labels = {name: torch.tensor([t.label for t in partitions[name]]) for name in ("support", "retain")}
    teacher = predict_logits(model, waves["retain"], int(settings.batch_size))
    torch.save(teacher, output / "teacher_logits.pt")
    threshold = float(json.loads((Path(settings.reference_run) / "selection.json").read_text())["threshold"])
    before = selection_metrics(model, waves, partitions, int(settings.batch_size), threshold)
    baseline_path = str((Path(settings.checkpoint) / "model.safetensors").resolve())
    selected: dict[str, Any] = {
        "status": "fitting",
        "threshold": threshold,
        "mode": str(settings.mode),
        "baseline_checkpoint": baseline_path,
        "guarded_checkpoint": baseline_path,
        "best_a21_checkpoint": baseline_path,
        "guarded_epoch": 0,
        "best_a21_epoch": 0,
        "baseline_metrics": before,
    }
    best_key = (float("inf"), float("inf"))
    guarded_key = (before["tune_eer"], before["tune_min_dcf"])
    history = [{"epoch": 0, "admissible": True, **before}]
    epochs = 1 if settings.stage == "smoke" else int(settings.epochs)
    for epoch in range(1, epochs + 1):
        updates = train_epoch(
            model,
            waves["support"],
            labels["support"],
            waves["retain"],
            labels["retain"],
            teacher,
            mode=str(settings.mode),
            learning_rate=float(settings.learning_rate),
            replay_weight=float(settings.replay_weight),
            beta=float(settings.distillation_weight),
            temperature=float(settings.temperature),
            batch_size=int(settings.train_batch_size),
            effective_batch_size=int(settings.train_batch_size) * int(settings.accumulation_steps),
            seed=int(cfg.seed) + epoch,
            max_updates=int(settings.smoke_steps) if settings.stage == "smoke" else None,
        )
        measured = selection_metrics(model, waves, partitions, int(settings.batch_size), threshold)
        allowed = admissible(measured, before, settings) and all(
            eer <= before["dev_generator_eer"][attack] + float(settings.max_generator_eer_degradation)
            for attack, eer in measured["dev_generator_eer"].items()
        )
        history.append({"epoch": epoch, "admissible": allowed, "updates": updates, **measured})
        write_json(output / "training_history.json", history)
        if settings.stage == "smoke":
            write_json(output / "smoke.json", {"status": "smoke_passed", "history": history})
            return
        key = (measured["tune_eer"], measured["tune_min_dcf"])
        for name, improved in (("best_a21", key < best_key), ("guarded", allowed and key < guarded_key)):
            if improved:
                path = output / f"{name}.safetensors"
                save_weights(model, path)
                selected[f"{name}_checkpoint"] = str(path.resolve())
                selected[f"{name}_epoch"] = epoch
        best_key = min(best_key, key)
        if allowed:
            guarded_key = min(guarded_key, key)
        log.info("Epoch %s: admissible=%s, tune EER=%s", epoch, allowed, measured["tune_eer"])
    selected["status"] = "complete"
    selected["checkpoint_sha256"] = {
        name: digest(Path(selected[f"{name}_checkpoint"])) for name in ("baseline", "best_a21", "guarded")
    }
    write_json(output / "selection.json", selected)


@hydra.main(version_base="1.3", config_path="../../configs", config_name="main")
def main(cfg: DictConfig) -> None:
    """Prepare by default; fitting requires an explicit smoke or fit stage.

    Args:
        cfg: Hydra configuration with steering=retention_a21.
    """
    settings = cfg.steering
    if settings.stage not in {"prepare", "smoke", "fit"}:
        message = "stage must be prepare, smoke or fit"
        raise ValueError(message)
    if settings.mode not in {"target", "replay", "projected"} or int(settings.epochs) < 1:
        message = "Invalid adaptation mode or epoch budget"
        raise ValueError(message)
    if bool(settings.bf16) or bool(settings.gradient_checkpointing):
        message = "This experiment uses FP32 and deterministic eval-mode backward"
        raise ValueError(message)
    output = Path(cfg.paths.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / "plan.json").exists():
            message = "Use a fresh run directory; partial training is not resumed"
            raise FileExistsError(message)
        partitions = prepare(cfg, output)
        if settings.stage == "prepare":
            log.info("Prepared retention experiment without model construction: %s", output)
            return
        set_seed(int(cfg.seed))
        torch.set_num_threads(4)
        fit(cfg, partitions, output)


if __name__ == "__main__":
    main()
