"""CPU tests for ALM masks, cleanup, and checkpoint-metric separation."""

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from src.alm_steering.checkpoints import read_log
from src.alm_steering.interventions import Intervention, prefix_mask, steer


@pytest.mark.parametrize(
    "scope,indices",
    [
        ("audio", [2, 3, 4]),
        ("text", [1, 5, 6]),
        ("both", [1, 2, 3, 4, 5, 6]),
    ],
)
def test_prefix_mask(scope: str, indices: list[int]) -> None:
    """Verify explicit positions and exclusion of all answer tokens.

    Args:
        scope: Requested position subset.
        indices: Expected selected positions.
    """
    assert prefix_mask(10, 7, (2, 5), scope).nonzero().flatten().tolist() == indices


def test_zero_identity_and_answer_preservation() -> None:
    """Zero is exact identity and active steering leaves BOS/answers untouched."""
    hidden = torch.randn(2, 10, 4)
    intervention = Intervention(torch.ones(4), 0, "both", 7, (2, 5))
    assert intervention.apply(hidden) is hidden
    intervention.strength = 2
    result = intervention.apply(hidden)
    torch.testing.assert_close(result[:, 1:7], hidden[:, 1:7] + 2)
    torch.testing.assert_close(result[:, 7:], hidden[:, 7:], rtol=0, atol=0)
    torch.testing.assert_close(result[:, 0], hidden[:, 0], rtol=0, atol=0)


def test_rank_one_and_cleanup() -> None:
    """Transport uses the source projection and hooks clean up after exceptions."""
    block = nn.Identity()
    hidden = torch.ones(2, 10, 4)
    intervention = Intervention(torch.ones(4), 1, "audio", 7, (2, 5), torch.zeros(4), torch.ones(4))
    with steer(block, intervention):
        result = block(hidden)
        torch.testing.assert_close(result[:, 2:5], torch.full((2, 3, 4), 5.0))
    torch.testing.assert_close(block(hidden), hidden, rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="sentinel"), steer(block, intervention):
        msg = "sentinel"
        raise RuntimeError(msg)
    torch.testing.assert_close(block(hidden), hidden, rtol=0, atol=0)


def test_invalid_boundaries() -> None:
    """Reject ambiguous scopes or spans instead of silently steering answers."""
    with pytest.raises(ValueError):
        prefix_mask(10, 7, (2, 8), "audio")
    with pytest.raises(ValueError):
        prefix_mask(10, 7, (2, 5), "answer")


def test_checkpoint_log_metric_separation(tmp_path: Path) -> None:
    """Keep explicit class accuracy separate from old token aggregate accuracy.

    Args:
        tmp_path: Test-owned temporary directory.
    """
    path = tmp_path / "log.txt"
    config = {"model": {"lora_rank": 8}}
    metrics = {"valid_agg_metrics": 0.77, "valid_classification_acc": 0.77, "valid_token_acc": 0.96}
    path.write_text(json.dumps(config, indent=2) + "\n" + json.dumps(metrics) + "\n")
    restored, records = read_log(path)
    assert restored == config
    assert records == [metrics]
