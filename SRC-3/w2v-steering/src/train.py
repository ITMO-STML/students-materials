"""Hydra entrypoint for antispoofing training."""

import logging
import os
from pathlib import Path

import hydra
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from transformers import TrainingArguments, set_seed

from .trainer import build_trainer

log = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def checkpoint_path(ckpt_path: str | None) -> str | None:
    """Return the resume checkpoint, or None to train from scratch.

    Args:
        ckpt_path: Checkpoint directory from the config. Blank starts a new run.

    Returns:
        Stripped checkpoint path, or None when `ckpt_path` is empty.

    """
    if ckpt_path is None:
        return None
    path = str(ckpt_path).strip()
    if not path:
        return None
    return path


def configure_trackio_dir() -> Path:
    """Point Trackio at a subfolder of the current Hydra run directory.

    Returns:
        Directory where this run's Trackio logs are written.

    """
    load_dotenv(PROJECT_ROOT / ".env")
    subdir = os.environ.get("TRACKIO_SUBDIR", "trackio")
    directory = Path(HydraConfig.get().runtime.output_dir) / subdir
    directory.mkdir(parents=True, exist_ok=True)
    os.environ["TRACKIO_DIR"] = str(directory.resolve())
    return directory


@hydra.main(version_base="1.3", config_path="../configs", config_name="main")
def main(cfg: DictConfig) -> None:
    """Seed, build the model and data, and train.

    Args:
        cfg: Composed Hydra config.

    """
    set_seed(int(cfg.seed))
    trackio_dir = configure_trackio_dir()
    model = hydra.utils.instantiate(cfg.model)
    data = hydra.utils.instantiate(cfg.data)
    raw_arguments = OmegaConf.to_container(cfg.training_arguments, resolve=True)
    if not isinstance(raw_arguments, dict):
        message = "training_arguments must be a mapping"
        raise TypeError(message)
    args = TrainingArguments(**raw_arguments)
    trainer = build_trainer(model, data, args)
    resume_from_checkpoint = checkpoint_path(cfg.ckpt_path)
    if resume_from_checkpoint is None:
        log.info("Starting training from scratch")
    else:
        log.info("Resuming training from %s", resume_from_checkpoint)
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    metrics = trainer.evaluate()
    log.info("Evaluation metrics: %s", metrics)
    log.info("Trackio logs: %s", trackio_dir)
    log.info("Finished training in %s", args.output_dir)


if __name__ == "__main__":
    main()
