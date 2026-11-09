"""Single configurable CLI for wav2vec2-AASIST evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import hydra

from src.evaluation import evaluate

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from torch import nn


def require_checkpoint(ckpt_path: str | None) -> str:
    """Retain the existing checkpoint helper import for downstream callers.

    Args:
        ckpt_path: Required checkpoint directory.

    Returns:
        Validated checkpoint directory string.
    """
    from src.evaluation.checkpoint import require_checkpoint as require

    return require(ckpt_path)


def load_checkpoint(model: nn.Module, ckpt_path: str) -> None:
    """Retain the existing checkpoint-loading API.

    Args:
        model: Model receiving saved parameters.
        ckpt_path: Checkpoint directory.
    """
    from src.evaluation.checkpoint import load_checkpoint as load

    load(model, ckpt_path)


@hydra.main(version_base="1.3", config_path=str(Path(__file__).resolve().parents[1] / "configs"), config_name="eval")
def main(cfg: DictConfig) -> None:
    """Dispatch the evaluation preset through the shared API.

    Args:
        cfg: Config selected with evaluation=<preset> and ordinary overrides.
    """
    evaluate(cfg)


if __name__ == "__main__":
    main()
