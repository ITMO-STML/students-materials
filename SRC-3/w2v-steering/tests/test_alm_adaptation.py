"""CPU checks for supervised ALM loss and exact gradient accumulation."""

import pytest
import torch
from torch import Tensor, nn

from src.alm_adaptation.training import answer_loss, train_epoch


def test_answer_loss_ignores_padding() -> None:
    """Padded logits neither change the loss nor receive gradients."""
    logits = torch.randn(2, 3, 5, requires_grad=True)
    targets = torch.tensor([[1, 2, 3], [2, 1, 0]])
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    result = answer_loss(logits, targets, mask)
    expected = torch.stack(
        [
            nn.functional.cross_entropy(logits[0], targets[0]),
            nn.functional.cross_entropy(logits[1, :2], targets[1, :2]),
        ],
    )
    torch.testing.assert_close(result, expected)
    result.sum().backward()
    assert torch.count_nonzero(logits.grad[1, 2]) == 0
    with pytest.raises(ValueError, match="nonempty"):
        answer_loss(logits, targets, torch.zeros_like(mask))


def test_accumulation_matches_full_batch() -> None:
    """Microbatch accumulation matches full updates, including partial groups."""
    torch.manual_seed(123)
    audio = torch.randn(11, 3)
    labels = [0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0]
    large, small = nn.Linear(3, 2), nn.Linear(3, 2)
    small.load_state_dict(large.state_dict())

    def run(model: nn.Module, microbatch: int) -> dict[str, float | int]:
        """Fit a tiny classifier through the production accumulation loop.

        Args:
            model: Test classifier.
            microbatch: Forward batch size.

        Returns:
            Training counters and mean loss.
        """

        def loss(features: Tensor, targets: list[int]) -> Tensor:
            """Return per-recording cross entropy.

            Args:
                features: Test features.
                targets: Binary class IDs.

            Returns:
                Per-recording loss vector.
            """
            return nn.functional.cross_entropy(model(features), torch.tensor(targets), reduction="none")

        return train_epoch(
            audio,
            labels,
            loss,
            list(model.parameters()),
            torch.optim.SGD(model.parameters(), lr=0.1),
            torch.amp.GradScaler("cuda", enabled=False),
            seed=42,
            effective_batch=8,
            microbatch=microbatch,
        )

    left, right = run(large, 8), run(small, 2)
    assert left["examples"] == right["examples"] == 11
    assert left["updates"] == right["updates"] == 2
    assert left["loss"] == pytest.approx(right["loss"], abs=1e-6)
    for first, second in zip(large.parameters(), small.parameters(), strict=True):
        torch.testing.assert_close(first, second)
