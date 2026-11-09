"""Memory-bounded, separately accumulated gradients for detector adaptation."""

import torch
from torch import Tensor, nn

from .gradients import combine_gradients, retention_loss


def predict_logits(model: nn.Module, waves: Tensor, batch_size: int) -> Tensor:
    """Cache fixed-teacher logits without keeping a second detector in memory.

    Args:
        model: Baseline detector before any parameter update.
        waves: Ordered CPU waveforms.
        batch_size: Inference microbatch size.

    Returns:
        Detached CPU logits aligned with the input waveforms.
    """
    model.eval()
    device = next(model.parameters()).device
    outputs = []
    with torch.no_grad():
        for batch in waves.split(batch_size):
            inputs = batch.to(device)
            outputs.append(
                model(input_values=inputs, attention_mask=torch.ones_like(inputs, dtype=torch.long))["logits"]
                .float()
                .cpu(),
            )
    result = torch.cat(outputs)
    if not torch.isfinite(result).all():
        message = "Nonfinite teacher logits"
        raise ValueError(message)
    return result


def gradients(
    model: nn.Module,
    waves: Tensor,
    labels: Tensor,
    batch_size: int,
    teacher: Tensor | None = None,
    beta: float = 1.0,
    temperature: float = 2.0,
) -> tuple[list[Tensor], float]:
    """Accumulate the exact mean gradient, including a partial microbatch.

    Args:
        model: Detector in deterministic evaluation mode, with autograd enabled.
        waves: CPU effective batch.
        labels: Aligned CPU labels.
        batch_size: Forward/backward microbatch size.
        teacher: Optional aligned baseline logits for retention loss.
        beta: Distillation coefficient.
        temperature: Distillation temperature.

    Returns:
        Detached per-parameter gradients and mean objective value.
    """
    if batch_size < 1 or len(waves) == 0 or len(waves) != len(labels):
        message = "Require nonempty aligned batches and positive microbatch size"
        raise ValueError(message)
    if teacher is not None and len(teacher) != len(waves):
        message = "Teacher cache is not aligned"
        raise ValueError(message)
    model.eval()  # Freeze dropout and BatchNorm buffers; parameters still receive gradients.
    model.zero_grad(set_to_none=True)
    device = next(model.parameters()).device
    total = 0.0
    for start in range(0, len(waves), batch_size):
        end = start + batch_size
        inputs, targets = waves[start:end].to(device), labels[start:end].to(device)
        logits = model(input_values=inputs, attention_mask=torch.ones_like(inputs, dtype=torch.long))["logits"]
        loss = (
            nn.functional.cross_entropy(logits.float(), targets)
            if teacher is None
            else retention_loss(logits, targets, teacher[start:end].to(device), beta, temperature)
        )
        if not torch.isfinite(loss):
            message = "Nonfinite adaptation loss"
            raise ValueError(message)
        fraction = len(inputs) / len(waves)
        (loss * fraction).backward()
        total += float(loss.detach()) * fraction
    result = [torch.zeros_like(p) if p.grad is None else p.grad.detach() for p in model.parameters() if p.requires_grad]
    model.zero_grad(set_to_none=True)
    return result, total


def train_epoch(
    model: nn.Module,
    target_waves: Tensor,
    target_labels: Tensor,
    retain_waves: Tensor,
    retain_labels: Tensor,
    teacher: Tensor,
    *,
    mode: str,
    learning_rate: float,
    replay_weight: float,
    beta: float,
    temperature: float,
    batch_size: int,
    effective_batch_size: int,
    seed: int,
    max_updates: int | None = None,
) -> list[dict[str, float]]:
    """Perform plain SGD so the projected direction is the actual update.

    Args:
        model: Trainable detector; stochastic layers and buffers remain frozen.
        target_waves: A21 plus genuine support waveforms.
        target_labels: Support labels.
        retain_waves: Training-only, generator-balanced replay pool.
        retain_labels: Replay labels.
        teacher: Baseline logits for the complete replay pool.
        mode: target, replay, or projected.
        learning_rate: Positive SGD step size; no momentum or weight decay.
        replay_weight: Retention-gradient coefficient.
        beta: KL coefficient inside the retention objective.
        temperature: Distillation temperature.
        batch_size: Microbatch size.
        effective_batch_size: Examples per target and replay gradient.
        seed: Local shuffle and replay-sampling seed.
        max_updates: Optional smoke budget.

    Returns:
        Per-update losses, sample counts, norms and conflict diagnostics.
    """
    if effective_batch_size < 1 or learning_rate <= 0 or len(retain_waves) == 0:
        message = "Require positive batch size, learning rate and replay pool"
        raise ValueError(message)
    if len(retain_labels) != len(retain_waves) or len(teacher) != len(retain_waves):
        message = "Replay tensors are not aligned"
        raise ValueError(message)
    if max_updates is not None and max_updates < 1:
        message = "max_updates must be positive"
        raise ValueError(message)
    generator = torch.Generator().manual_seed(seed)
    batches = torch.randperm(len(target_waves), generator=generator).split(effective_batch_size)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(parameters, lr=learning_rate)
    history = []
    for indices in batches[:max_updates]:
        replay = torch.randint(len(retain_waves), (len(indices),), generator=generator)
        target, target_loss = gradients(model, target_waves[indices], target_labels[indices], batch_size)
        retain, retain_value = gradients(
            model,
            retain_waves[replay],
            retain_labels[replay],
            batch_size,
            teacher[replay],
            beta,
            temperature,
        )
        direction, diagnostics = combine_gradients(target, retain, mode, replay_weight)
        for parameter, gradient in zip(parameters, direction, strict=True):
            parameter.grad = gradient
        norm = nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        history.append(
            {
                "target_loss": target_loss,
                "retain_loss": retain_value,
                "target_examples": float(len(indices)),
                "retain_examples": float(len(replay)),
                "gradient_norm": float(norm),
                **diagnostics,
            },
        )
    if not history:
        message = "Target support must not be empty"
        raise ValueError(message)
    return history
