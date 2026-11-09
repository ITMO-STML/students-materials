"""Small supervised answer-loss and exact accumulation primitives."""

from collections.abc import Callable

import torch
from torch import Tensor, nn


def answer_loss(logits: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
    """Average answer-token cross entropy separately for each recording.

    Args:
        logits: Causally aligned batch, answer length, vocabulary logits.
        targets: Answer token IDs.
        mask: Valid answer positions, excluding padding.

    Returns:
        One differentiable mean loss per recording.
    """
    if logits.shape[:2] != targets.shape or targets.shape != mask.shape or not (mask.sum(-1) > 0).all():
        message = "Require aligned answer logits and nonempty masks"
        raise ValueError(message)
    values = nn.functional.cross_entropy(logits.float().transpose(1, 2), targets, reduction="none")
    return (values * mask).sum(-1) / mask.sum(-1)


def train_epoch(
    audio: Tensor,
    labels: list[int],
    loss_fn: Callable[[Tensor, list[int]], Tensor],
    parameters: list[nn.Parameter],
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    seed: int,
    effective_batch: int = 8,
    microbatch: int = 2,
    max_updates: int | None = None,
) -> dict[str, float | int]:
    """Fit shuffled support with sample-exact accumulation and finite gradients.

    Args:
        audio: Cached support embeddings or test features.
        labels: Corresponding class labels.
        loss_fn: Returns one loss per recording.
        parameters: Only parameters authorized for training.
        optimizer: Optimizer over those parameters.
        scaler: CUDA gradient scaler, disabled for CPU tests.
        seed: Independent deterministic shuffle seed.
        effective_batch: Recordings per optimizer update.
        microbatch: Recordings per forward/backward pass.
        max_updates: Optional bounded smoke-test update count.

    Returns:
        Mean loss, examples and update count.
    """
    if len(audio) != len(labels) or not labels or min(effective_batch, microbatch) < 1:
        message = "Require aligned nonempty inputs and positive batch sizes"
        raise ValueError(message)
    if max_updates is not None and max_updates < 1:
        message = "max_updates must be positive"
        raise ValueError(message)
    order = torch.randperm(len(audio), generator=torch.Generator().manual_seed(seed))
    groups = list(order.split(effective_batch))[:max_updates]
    total, examples = 0.0, 0
    for group in groups:
        optimizer.zero_grad(set_to_none=True)
        for indices in group.split(microbatch):
            losses = loss_fn(audio[indices], [labels[i] for i in indices.tolist()])
            if losses.shape != (len(indices),) or not torch.isfinite(losses).all():
                message = "Expected finite per-recording supervised losses"
                raise ValueError(message)
            scaler.scale(losses.sum() / len(group)).backward()
            total += float(losses.detach().sum())
            examples += len(indices)
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()
    return {"loss": total / examples, "examples": examples, "updates": len(groups)}
