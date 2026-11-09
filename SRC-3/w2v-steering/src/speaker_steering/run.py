"""Locked Hydra entry point for speaker fitting and full-dataset evaluation."""

import fcntl
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from transformers import set_seed

from src.speaker_steering.adaptation import fit
from src.speaker_steering.evaluation import evaluate


@hydra.main(version_base="1.3", config_path="../../configs", config_name="main")
def main(cfg: DictConfig) -> None:
    """Freeze a support-only adaptation or evaluate its immutable selection.

    Args:
        cfg: Configuration using steering=elevenlabs_speaker.
    """
    set_seed(int(cfg.seed))
    torch.set_num_threads(4)
    output = Path(cfg.steering.resume_dir or cfg.paths.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if cfg.steering.stage == "fit":
            fit(cfg, output)
        elif cfg.steering.stage == "evaluate":
            evaluate(cfg, output)
        else:
            message = f"Unknown stage: {cfg.steering.stage}"
            raise ValueError(message)


if __name__ == "__main__":
    main()
