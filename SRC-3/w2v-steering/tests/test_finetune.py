"""Supervised adaptation fitting and gradient accumulation checks."""

from typing import override

import pytest
import torch
from torch import Tensor, nn

from src.steering.finetune import train_epoch


class SmallClassifier(nn.Module):
    """Minimal two-class detector with the production forward signature."""

    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Linear(2, 2)

    @override
    def forward(self, input_values: Tensor, attention_mask: Tensor, labels: Tensor) -> dict[str, Tensor]:
        """Compute supervised loss on masked features.

        Args:
            input_values: Two input features per example.
            attention_mask: Valid input indicators.
            labels: Class indices.

        Returns:
            Cross-entropy loss.
        """
        logits = self.head(input_values * attention_mask)
        return {"loss": nn.functional.cross_entropy(logits, labels)}


def test_accumulation_matches_large_batches_with_partial_group() -> None:
    """Microbatching matches full batches, including the final partial group."""
    torch.manual_seed(17)
    torch.set_num_threads(2)
    large, small = SmallClassifier(), SmallClassifier()
    small.load_state_dict(large.state_dict())
    waves, labels = torch.randn(7, 2), torch.tensor([0, 1, 0, 1, 0, 1, 1])
    left = train_epoch(large, waves, labels, torch.optim.SGD(large.parameters(), lr=0.1), 6, 1, 42, bf16=False)
    right = train_epoch(small, waves, labels, torch.optim.SGD(small.parameters(), lr=0.1), 2, 3, 42, bf16=False)
    assert left["updates"] == right["updates"] == 2
    assert left["examples"] == right["examples"] == 7
    assert left["loss"] == pytest.approx(right["loss"], abs=1e-6)
    for first, second in zip(large.parameters(), small.parameters(), strict=True):
        torch.testing.assert_close(first, second)


def test_fitting_smoke_reduces_loss_and_honors_budget() -> None:
    """Small supervised fits learn separable classes and cap optimizer updates."""
    torch.manual_seed(18)
    model = SmallClassifier()
    waves = torch.tensor([[-2.0, -1.0], [2.0, 1.0]]).repeat(8, 1)
    labels = torch.tensor([0, 1]).repeat(8)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
    first = train_epoch(model, waves, labels, optimizer, 2, 2, 42, bf16=False, max_updates=2)
    for epoch in range(5):
        last = train_epoch(model, waves, labels, optimizer, 2, 2, 43 + epoch, bf16=False)
    assert first["updates"] == 2
    assert first["examples"] == 8
    assert last["loss"] < first["loss"]
    assert model.training is False
    with pytest.raises(ValueError, match="max_updates"):
        train_epoch(model, waves, labels, optimizer, 2, 2, 42, bf16=False, max_updates=0)
