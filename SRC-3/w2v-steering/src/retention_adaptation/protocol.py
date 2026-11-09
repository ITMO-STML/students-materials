"""Training-only retention replay alongside the frozen A21 pilot cohorts."""

import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from src.steering.protocol import Trial, check_disjoint, read_trials, select_trials


def digest(path: Path) -> str:
    """Hash a local artifact without loading its tensors.

    Args:
        path: Existing file.

    Returns:
        SHA-256 hexadecimal digest.
    """
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def replay_pool(trials: list[Trial], per_generator: int, seed: int) -> list[Trial]:
    """Select equal spoof-generator counts and equal total genuine/spoof counts.

    Args:
        trials: Official ASVspoof training metadata, never dev/eval.
        per_generator: Unique recordings from each non-A21 generator.
        seed: Deterministic sampling seed.

    Returns:
        Class-balanced replay pool, uniform over spoof generators.
    """
    attacks = sorted({t.attack for t in trials if t.label == 1 and t.attack != "A21"})
    if not attacks:
        message = "No non-A21 retention generators"
        raise ValueError(message)
    return select_trials(trials, ["bonafide"], per_generator * len(attacks), seed) + select_trials(
        trials,
        attacks,
        per_generator,
        seed,
    )


def prepare(cfg: DictConfig, output: Path) -> dict[str, list[Trial]]:
    """Freeze manifests, audit leakage and verify the baseline without inference.

    Args:
        cfg: Retention experiment configuration.
        output: New experiment directory.

    Returns:
        Support/tune/dev/test plus training-only retention metadata.
    """
    reference = Path(cfg.steering.reference_run)
    raw = json.loads((reference / "manifest.json").read_text())
    partitions = {name: [Trial(**row) for row in raw[name]] for name in ("support", "tune", "dev", "test")}
    if {t.attack for t in partitions["support"]} != {"bonafide", "A21"}:
        message = "Expected the original A21 plus bonafide support"
        raise ValueError(message)
    partitions["retain"] = replay_pool(
        read_trials(Path(cfg.data.root), "train"),
        int(cfg.steering.retain_per_generator),
        int(cfg.seed),
    )
    check_disjoint(partitions)
    for trials in partitions.values():
        for trial in trials:
            path = Path(trial.audio_dir) / f"{trial.utterance}.flac"
            if not path.is_file():
                raise FileNotFoundError(path)
    checkpoint = Path(cfg.steering.checkpoint) / "model.safetensors"
    checkpoint_hash = digest(checkpoint)
    if checkpoint_hash != json.loads((reference / "provenance.json").read_text())["checkpoint_sha256"]:
        message = "Baseline differs from the frozen A21 reference"
        raise ValueError(message)
    manifest = {name: [asdict(t) for t in trials] for name, trials in partitions.items()}
    plan: dict[str, Any] = {
        "status": "prepared",
        "checkpoint_sha256": checkpoint_hash,
        "reference_run": str(reference),
        "reference_manifest_sha256": digest(reference / "manifest.json"),
        "reference_selection_sha256": digest(reference / "selection.json"),
        "manifest_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
        "counts": {name: dict(Counter(t.attack for t in rows)) for name, rows in partitions.items()},
        "config": OmegaConf.to_container(cfg, resolve=True),
        "sources_sha256": {str(p): digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        "note": (
            "A21 support/tune originate from eval; only the frozen test cohort is held out. Replay uses train only."
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    for name, payload in (("manifest.json", manifest), ("plan.json", plan)):
        (output / name).write_text(json.dumps(payload, indent=2) + "\n")
    return partitions
