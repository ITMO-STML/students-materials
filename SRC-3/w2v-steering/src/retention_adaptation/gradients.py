"""CLEAR-inspired retention objectives and asymmetric gradient projection."""

import math

import torch
from torch import Tensor
from torch.nn import functional as F


def retention_loss(logits: Tensor, labels: Tensor, teacher: Tensor, beta: float, temperature: float) -> Tensor:
    """Combine supervised replay with forward teacher-to-student KL.

    Args:
        logits: Current two-class logits.
        labels: Replay ground truth, zero for genuine audio.
        teacher: Frozen baseline logits for exactly these recordings.
        beta: Nonnegative distillation weight.
        temperature: Positive distillation temperature.

    Returns:
        Mean replay cross-entropy plus temperature-scaled KL.
    """
    if not math.isfinite(beta) or beta < 0 or not math.isfinite(temperature) or temperature <= 0:
        message = "Require finite beta >= 0 and temperature > 0"
        raise ValueError(message)
    if logits.shape != teacher.shape:
        message = "Student and teacher logits must have identical shapes"
        raise ValueError(message)
    kl = F.kl_div(
        F.log_softmax(logits.float() / temperature, dim=-1),
        F.softmax(teacher.detach().float() / temperature, dim=-1),
        reduction="batchmean",
    )
    return F.cross_entropy(logits.float(), labels) + beta * temperature**2 * kl


def dot(left: list[Tensor], right: list[Tensor]) -> Tensor:
    """Compute a global parameter-space inner product without concatenation.

    Args:
        left: First gradient, split by parameter.
        right: Second gradient with matching shapes.

    Returns:
        Scalar accumulated in double precision.
    """
    return torch.stack([(a * b).sum(dtype=torch.float64) for a, b in zip(left, right, strict=True)]).sum()


def combine_gradients(
    target: list[Tensor],
    retain: list[Tensor],
    mode: str,
    weight: float,
) -> tuple[list[Tensor], dict[str, float]]:
    """Combine separately accumulated target and retention gradients.

    Args:
        target: Gradient of target CE over one effective batch.
        retain: Gradient of retention CE plus KL over one effective batch.
        mode: target, replay, or projected.
        weight: Nonnegative replay coefficient.

    Returns:
        Update direction and conflict diagnostics before gradient clipping.
    """
    if mode not in {"target", "replay", "projected"} or not math.isfinite(weight) or weight < 0:
        message = "Invalid gradient mode or replay weight"
        raise ValueError(message)
    if not target or len(target) != len(retain) or any(a.shape != b.shape for a, b in zip(target, retain, strict=True)):
        message = "Require nonempty aligned gradients"
        raise ValueError(message)
    cross, norm = dot(target, retain), dot(retain, retain)
    target_norm = dot(target, target)
    if not torch.isfinite(torch.stack([cross, norm, target_norm])).all():
        message = "Nonfinite gradient"
        raise ValueError(message)
    conflict = float(cross) < 0
    coefficient = float(cross / norm) if mode == "projected" and conflict and float(norm) > 0 else 0.0
    replay = 0.0 if mode == "target" else weight
    direction = [a + (replay - coefficient) * b for a, b in zip(target, retain, strict=True)]
    denominator = float((target_norm * norm).sqrt())
    return direction, {
        "conflict": float(conflict),
        "cosine": float(cross) / denominator if denominator > 0 else 0.0,
        "target_retain_dot": float(cross),
        "update_retain_dot": float(dot(direction, retain)),
        "projection_coefficient": coefficient,
    }
