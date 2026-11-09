"""Checkpoint path handling for the training and evaluation entrypoints."""

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from src.eval import load_checkpoint, require_checkpoint
from src.train import checkpoint_path


def test_empty_ckpt_path_starts_from_scratch() -> None:
    """A blank checkpoint path trains from the beginning."""
    assert checkpoint_path(None) is None
    assert checkpoint_path("") is None
    assert checkpoint_path("   ") is None


def test_ckpt_path_resumes() -> None:
    """A set checkpoint path is used to resume training."""
    assert checkpoint_path(" logs/runs/checkpoint-300 ") == "logs/runs/checkpoint-300"


def test_eval_requires_checkpoint() -> None:
    """Evaluation rejects a blank checkpoint path."""
    with pytest.raises(ValueError, match="ckpt_path is required"):
        require_checkpoint("")
    assert require_checkpoint(" logs/runs/checkpoint-300 ") == "logs/runs/checkpoint-300"


def test_load_checkpoint(tmp_path: Path) -> None:
    """Trainer safetensors weights replace the module parameters."""
    source = nn.Linear(2, 1)
    destination = nn.Linear(2, 1)
    save_file(source.state_dict(), tmp_path / "model.safetensors")
    load_checkpoint(destination, str(tmp_path))
    assert torch.equal(destination.weight, source.weight)
    assert torch.equal(destination.bias, source.bias)
