"""Hydra entrypoint for auditable SALMONN steering experiments."""

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

from src.alm_steering.interventions import Intervention, steer
from src.alm_steering.runtime import FrozenALM
from src.steering.protocol import Trial


def write_json(path: Path, value: Any) -> None:
    """Atomically save strict JSON.

    Args:
        path: Output destination.
        value: Serializable experiment state.
    """
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def smoke(model: FrozenALM, cfg: DictConfig, output: Path) -> None:
    """Check loading, prefix identity, frozen weights, and zero-strength scores.

    Args:
        model: Restored ALM.
        cfg: Hydra configuration.
        output: Run artifact directory.
    """
    manifest = json.loads(Path(cfg.alm.manifest).read_text())
    selected = [
        Trial(**row) for label in (0, 1) for row in [item for item in manifest["support"] if item["label"] == label][:4]
    ]
    started = time.perf_counter()
    embeddings = [
        model.audio_embeddings(selected[index : index + int(cfg.alm.batch_size)])
        for index in range(0, len(selected), int(cfg.alm.batch_size))
    ]
    audio = torch.cat(embeddings)
    extraction_seconds = time.perf_counter() - started
    torch.save(audio, output / "smoke_audio.pt")
    batch = audio[: int(cfg.alm.batch_size)]
    with torch.no_grad(), model.model.maybe_autocast():
        prefix, span = model.prefix(batch)
        wrapped, _ = model.model.prompt_wrap(
            batch.to(model.device),
            torch.ones(batch.shape[:2], dtype=torch.long, device=model.device),
            model.prompt,
        )
        torch.testing.assert_close(prefix[:, 1:], wrapped, rtol=0, atol=0)
    scores = model.score(batch)
    intervention = Intervention(torch.ones(prefix.shape[-1]), 0.0, "audio", prefix.shape[1], span)
    with steer(model.layers[7], intervention):
        zero = model.score(batch)
    np.testing.assert_array_equal(scores, zero)
    assert not any(parameter.requires_grad for parameter in model.model.parameters())
    np.testing.assert_array_equal(model.score(batch), scores)
    all_scores = np.concatenate([model.score(audio[index : index + 4]) for index in range(0, len(audio), 4)])
    generated = model.greedy(audio[[0, 4]])
    report = {
        "status": "complete",
        "load": model.load_audit,
        "zero_strength_max_difference": float(np.max(np.abs(scores - zero))),
        "audio_shape": list(audio.shape),
        "extraction_seconds": extraction_seconds,
        "utterances": [trial.utterance for trial in selected],
        "labels": [trial.label for trial in selected],
        "class_log_odds": all_scores.tolist(),
        "generated_examples": generated,
        "note": "Support-only smoke diagnostics, not held-out or full evaluation metrics.",
    }
    write_json(output / "smoke.json", report)
    write_json(output / "progress.json", {"status": "complete", "stage": "smoke"})
    logging.getLogger(__name__).info("ALM smoke passed: %s", report)


@hydra.main(version_base="1.3", config_path="../../configs", config_name="main")
def main(cfg: DictConfig) -> None:
    """Restore the selected ALM and run the configured stage.

    Args:
        cfg: Main Hydra config with +alm=default.
    """
    torch.set_num_threads(4)
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    output = Path(cfg.alm.resume_dir or cfg.paths.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "progress.json", {"status": "loading", "stage": str(cfg.alm.stage)})
    settings = OmegaConf.to_container(cfg.alm, resolve=True)
    sources = [*Path("src/alm_steering").glob("*.py"), *Path(cfg.alm.source_package).glob("*.py")]
    write_json(
        output / "provenance.json",
        {
            "settings": settings,
            "seed": int(cfg.seed),
            "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
            "runtime": {"torch": torch.__version__},
        },
    )
    model = FrozenALM(settings)
    if cfg.alm.stage == "smoke":
        smoke(model, cfg, output)
    else:
        msg = f"Unsupported ALM stage: {cfg.alm.stage}"
        raise ValueError(msg)


if __name__ == "__main__":
    main()
