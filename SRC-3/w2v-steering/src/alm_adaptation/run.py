"""Queued, support-only LoRA comparator with tuning/development selection."""

import fcntl
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from src.alm_adaptation.training import answer_loss, train_epoch
from src.alm_steering.classification import classification_rank, classification_report, retention_allowed
from src.alm_steering.pilot import score_group
from src.alm_steering.run import write_json
from src.alm_steering.runtime import FrozenALM
from src.steering.protocol import Trial, check_disjoint

log = logging.getLogger(__name__)


def wait_for_reference(reference: Path, timeout: int) -> None:
    """Wait without allocating a GPU; stop if the prerequisite exits incomplete.

    Args:
        reference: Running steering pilot directory.
        timeout: Maximum queue duration in seconds.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        progress = json.loads((reference / "progress.json").read_text())
        if progress.get("stage") == "complete" and progress.get("status") == "complete":
            assert (reference / "report.json").exists()
            return
        # NFS requires a writable descriptor for an exclusive advisory lock.
        with (reference / "coordinator.lock").open("r+") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            else:
                message = "Steering pilot exited incomplete; inspect it before training"
                raise RuntimeError(message)
        log.info("Queued behind steering pilot: %s", progress)
        time.sleep(30)
    message = "Timed out waiting for the steering pilot"
    raise TimeoutError(message)


def load_audio(reference: Path, group: str, count: int) -> Tensor:
    """Read immutable projected-audio chunks from a completed pilot.

    Args:
        reference: Pilot run directory.
        group: Manifest group name.
        count: Expected number of recordings.

    Returns:
        Aligned CPU audio embeddings.
    """
    chunks = []
    for start in range(0, count, 32):
        end = min(start + 32, count)
        path = reference / f"audio_{group}" / f"{start:06d}_{end:06d}.pt"
        values = torch.load(path, weights_only=True, map_location="cpu")
        assert len(values) == end - start and torch.isfinite(values).all()
        chunks.append(values)
    return torch.cat(chunks)


def supervised_loss(model: FrozenALM, audio: Tensor, labels: list[int]) -> Tensor:
    """Compute class-answer cross entropy without reasoning or prefix targets.

    Args:
        model: ALM with only LoRA parameters trainable.
        audio: Cached audio embeddings.
        labels: Zero for bonafide, one for spoof.

    Returns:
        One mean answer-token loss per recording.
    """
    answers = [f"Final Answer: {'bonafide' if label == 0 else 'spoof'}</s>" for label in labels]
    targets = model.tokenizer(answers, padding=True, add_special_tokens=False, return_tensors="pt")
    ids, mask = targets.input_ids.to(model.device), targets.attention_mask.to(model.device)
    with model.model.maybe_autocast():
        prefix, _ = model.prefix(audio)
        inputs = torch.cat([prefix, model.embedding(ids)], dim=1)
        attention = torch.cat([torch.ones(prefix.shape[:2], device=model.device), mask], dim=1)
        logits = model.llm(inputs_embeds=inputs, attention_mask=attention, use_cache=False).logits
        return answer_loss(logits[:, prefix.shape[1] - 1 : -1], ids, mask)


def selection_allowed(metrics: dict[str, Any], baseline: dict[str, Any]) -> bool:
    """Apply the same prespecified retention constraints as the steering pilot.

    Args:
        metrics: Candidate tune/dev pooled metrics.
        baseline: Frozen reference tune/dev pooled metrics.

    Returns:
        Whether all retention constraints hold.
    """
    return metrics["tune"]["bonafide_recall"] >= baseline["tune"]["bonafide_recall"] - 0.05 and retention_allowed(
        metrics["dev"],
        baseline["dev"],
    )


def save_adapters(adapters: dict[str, torch.nn.Parameter], path: Path) -> None:
    """Atomically save only the approved LoRA parameters.

    Args:
        adapters: Named LoRA parameters.
        path: Run-owned destination.
    """
    temporary = path.with_suffix(".partial")
    torch.save({name: value.detach().cpu().clone() for name, value in adapters.items()}, temporary)
    temporary.replace(path)


def fit(
    model: FrozenALM,
    audio: dict[str, Tensor],
    trials: dict[str, list[Trial]],
    baseline: dict[str, Any],
    cfg: DictConfig,
    output: Path,
) -> dict[str, Any]:
    """Fit five support-only epochs and freeze a guarded selection before test.

    Args:
        model: Restored baseline model.
        audio: Non-test cached embeddings.
        trials: Manifest records.
        baseline: Reference threshold and selection metrics.
        cfg: Approved experiment settings.
        output: New run directory.

    Returns:
        Fixed checkpoint selection and per-epoch history.
    """
    settings = cfg.alm_finetune
    adapters = {name: value for name, value in model.model.named_parameters() if "lora_" in name}
    assert sum(value.numel() for value in adapters.values()) == 4194304
    for parameter in adapters.values():
        parameter.requires_grad_(True)
    assert all("lora_" in name for name, value in model.model.named_parameters() if value.requires_grad)
    optimizer = torch.optim.AdamW(
        adapters.values(),
        lr=float(settings.learning_rate),
        weight_decay=float(settings.weight_decay),
    )
    scaler = torch.amp.GradScaler("cuda")
    labels = [trial.label for trial in trials["support"]]
    selected_path = output / "selected_lora.pt"
    save_adapters(adapters, selected_path)
    best = {"epoch": 0, "admissible": True, "metrics": baseline["metrics"]}
    history = [best]
    for epoch in range(1, int(settings.epochs) + 1):
        write_json(output / "progress.json", {"stage": "training", "epoch": epoch})
        model.llm.train()
        started = time.perf_counter()
        trained = train_epoch(
            audio["support"],
            labels,
            lambda x, y: supervised_loss(model, x, y),
            list(adapters.values()),
            optimizer,
            scaler,
            int(cfg.seed) + epoch,
            int(settings.effective_batch),
            int(settings.microbatch),
        )
        trained["wall_seconds"] = time.perf_counter() - started
        model.model.eval()
        scores = {
            name: score_group(model, audio[name], output / f"epoch_{epoch}_{name}.json", int(cfg.alm.batch_size))
            for name in ("tune", "dev")
        }
        metrics = {name: classification_report(trials[name], scores[name])["pooled"] for name in scores}
        allowed = selection_allowed(metrics, baseline["metrics"])
        row = {"epoch": epoch, "admissible": allowed, "metrics": metrics, "training": trained}
        save_adapters(adapters, output / f"epoch_{epoch}_lora.pt")
        rank = classification_rank(metrics["tune"])
        previous = classification_rank(best["metrics"]["tune"])
        if allowed and rank < previous:
            best = row
            save_adapters(adapters, selected_path)
        history.append(row)
        write_json(output / "training_history.json", history)
        log.info("Epoch %d: %s", epoch, row)
    selection = {"selected": best, "history": history, "threshold": baseline["threshold"], "baseline_fallback": True}
    write_json(output / "selection.json", selection)
    state = torch.load(selected_path, weights_only=True, map_location="cpu")
    with torch.no_grad():
        for name, parameter in adapters.items():
            parameter.copy_(state[name])
            parameter.grad = None
    model.model.requires_grad_(False).eval()
    return selection


def evaluate(
    model: FrozenALM,
    trials: dict[str, list[Trial]],
    reference: Path,
    output: Path,
    selection: dict[str, Any],
    batch_size: int,
) -> None:
    """Evaluate the frozen choice and measure its cached-audio scoring time.

    Args:
        model: Selected frozen adapter.
        trials: Fixed pilot manifest.
        reference: Completed steering pilot.
        output: Comparator run directory.
        selection: Saved tune/dev-only decision.
        batch_size: Inference batch size.
    """
    write_json(output / "progress.json", {"stage": "pilot_test"})
    audio = load_audio(reference, "test", len(trials["test"]))
    scores = score_group(model, audio, output / "test_scores.json", batch_size)
    report = classification_report(trials["test"], scores)
    batch = audio[:batch_size]
    model.score(batch)
    torch.cuda.synchronize(model.device)
    torch.cuda.reset_peak_memory_stats(model.device)
    rounds = []
    for _ in range(5):
        started = time.perf_counter()
        model.score(batch)
        torch.cuda.synchronize(model.device)
        rounds.append(time.perf_counter() - started)
    write_json(
        output / "compute_benchmark.json",
        {
            "scope": "cached audio scoring only; excludes encoder, disk, fitting and loading",
            "batch_size": batch_size,
            "seconds": rounds,
            "mean_seconds": float(np.mean(rounds)),
            "std_seconds": float(np.std(rounds, ddof=1)),
            "peak_allocated_bytes_including_model": torch.cuda.max_memory_allocated(model.device),
        },
    )
    write_json(
        output / "report.json",
        {
            "cohort": "4352-recording historical pilot, NOT full evaluation",
            "method": "LoRA-only supervised comparator with guarded epoch selection and baseline fallback",
            "selected_epoch": selection["selected"]["epoch"],
            "threshold": selection["threshold"],
            "results": report,
        },
    )
    write_json(output / "progress.json", {"stage": "complete", "status": "complete"})


@hydra.main(version_base="1.3", config_path="../../configs", config_name="main")
def main(cfg: DictConfig) -> None:
    """Queue behind steering, verify provenance, fit and test the approved LoRA.

    Args:
        cfg: Hydra configuration with alm and alm_finetune groups.
    """
    output = Path(cfg.paths.output_dir)
    reference = Path(cfg.alm_finetune.reference_run)
    with (output / "coordinator.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / "training_history.json").exists():
            message = "Training already exists: use a new run directory instead of overwriting"
            raise FileExistsError(message)
        write_json(output / "progress.json", {"stage": "waiting_for_pilot", "reference": str(reference)})
        wait_for_reference(reference, int(cfg.alm_finetune.wait_seconds))
        identity = json.loads((reference / "identity.json").read_text())
        for filename, digest in identity["sha256"].items():
            assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == digest, filename
        assert json.loads((reference / "finetune_smoke.json").read_text())["status"] == "passed"
        settings = OmegaConf.to_container(cfg.alm, resolve=True)
        for key, value in identity["settings"].items():
            if key != "stage":
                assert settings[key] == value, key
        assert int(cfg.seed) == identity["seed"]
        write_json(
            output / "provenance.json",
            {
                "reference_identity": identity,
                "settings": OmegaConf.to_container(cfg.alm_finetune, resolve=True),
                "seed": int(cfg.seed),
                "source_sha256": {
                    str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in Path("src/alm_adaptation").glob("*.py")
                },
            },
        )
        torch.set_num_threads(4)
        torch.manual_seed(int(cfg.seed))
        np.random.seed(int(cfg.seed))
        write_json(output / "progress.json", {"stage": "loading"})
        model = FrozenALM(settings)
        raw = json.loads((reference / "manifest.json").read_text())
        trials = {name: [Trial(**row) for row in rows] for name, rows in raw.items()}
        check_disjoint(trials)
        assert len(trials["support"]) == 256 and {t.attack for t in trials["support"]} == {"A21", "bonafide"}
        write_json(output / "manifest.json", raw)
        audio = {name: load_audio(reference, name, len(trials[name])) for name in ("support", "tune", "dev")}
        baseline = {"threshold": 0.0, "metrics": {}}
        for name in ("tune", "dev"):
            actual = score_group(model, audio[name], output / f"baseline_{name}.json", int(cfg.alm.batch_size))
            expected = json.loads((reference / "scores" / f"baseline_{name}.json").read_text())["scores"]
            np.testing.assert_array_equal(actual, expected)
            baseline["metrics"][name] = classification_report(trials[name], actual)["pooled"]
        selection = fit(model, audio, trials, baseline, cfg, output)
        evaluate(model, trials, reference, output, selection, int(cfg.alm.batch_size))
        log.info("ALM LoRA comparator complete: %s", output)


if __name__ == "__main__":
    main()
