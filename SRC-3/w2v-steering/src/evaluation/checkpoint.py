"""Trainer checkpoint backend for the common evaluation API."""

import logging
import os
from pathlib import Path

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_file
from torch import nn
from transformers import TrainingArguments

log = logging.getLogger(__name__)


def require_checkpoint(ckpt_path: str | None) -> str:
    """Return the checkpoint directory required for evaluation.

    Args:
        ckpt_path: Checkpoint directory from the config. Blank is rejected.

    Returns:
        Stripped checkpoint path.

    Raises:
        ValueError: `ckpt_path` is empty.

    """
    from src.train import checkpoint_path

    path = checkpoint_path(ckpt_path)
    if path is None:
        message = "ckpt_path is required to evaluate a checkpoint"
        raise ValueError(message)
    return path


def load_checkpoint(model: nn.Module, ckpt_path: str) -> None:
    """Load Trainer weights into `model`.

    Args:
        model: Module whose parameters match the checkpoint.
        ckpt_path: Directory that contains `model.safetensors`.

    Raises:
        FileNotFoundError: The checkpoint directory has no `model.safetensors`.

    """
    weights = Path(ckpt_path) / "model.safetensors"
    if not weights.is_file():
        message = f"Checkpoint is missing model.safetensors: {ckpt_path}"
        raise FileNotFoundError(message)
    model.load_state_dict(load_file(weights))


def run(cfg: DictConfig, output: Path) -> None:
    """Seed, load a checkpoint, and score the ASVspoof 5 test split.

    Args:
        cfg: Composed config; ckpt_path points at a Trainer checkpoint.
        output: Locked artifact directory selected by the common launcher.

    """
    from src.trainer import build_trainer

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    trackio_dir = output / os.environ.get("TRACKIO_SUBDIR", "trackio")
    trackio_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TRACKIO_DIR"] = str(trackio_dir.resolve())
    ckpt = require_checkpoint(cfg.ckpt_path)
    model = hydra.utils.instantiate(cfg.model)
    load_checkpoint(model, ckpt)
    data = hydra.utils.instantiate(cfg.data)
    raw_arguments = OmegaConf.to_container(cfg.training_arguments, resolve=True)
    if not isinstance(raw_arguments, dict):
        message = "training_arguments must be a mapping"
        raise TypeError(message)
    raw_arguments["output_dir"] = str(output)
    args = TrainingArguments(**raw_arguments)
    trainer = build_trainer(model, data, args)
    log.info("Evaluating %s on the test split", ckpt)
    prediction = trainer.predict(data.test_dataset)
    log.info("Test metrics: %s", prediction.metrics)
    log.info("Trackio logs: %s", trackio_dir)
    log.info("Finished evaluation in %s", args.output_dir)
