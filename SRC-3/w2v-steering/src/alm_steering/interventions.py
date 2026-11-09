"""Prefix-only activation interventions for frozen ALMs."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn


def prefix_mask(length: int, prefix_length: int, audio_span: tuple[int, int], scope: str) -> Tensor:
    """Select prefix positions, excluding BOS and all answer tokens.

    Args:
        length: Actual sequence length including answers.
        prefix_length: Original prompt length.
        audio_span: Half-open audio-token span.
        scope: audio, text, or both.

    Returns:
        Boolean position mask.
    """
    start, end = audio_span
    if not 1 <= start < end <= prefix_length <= length:
        msg = "Invalid prefix/audio boundaries"
        raise ValueError(msg)
    if scope not in {"audio", "text", "both"}:
        msg = f"Unknown intervention scope: {scope}"
        raise ValueError(msg)
    mask = torch.zeros(length, dtype=torch.bool)
    if scope in {"text", "both"}:
        mask[1:prefix_length] = True
    if scope == "text":
        mask[start:end] = False
    else:
        mask[start:end] = True
    return mask


@dataclass
class Intervention:
    """Describe an additive or rank-one prefix correction.

    Args:
        vector: Additive direction or transport write vector.
        strength: Signed scale.
        scope: Prefix positions to change.
        prefix_length: Original prompt length including BOS.
        audio_span: Projected audio positions.
        anchor: Optional transport anchor.
        read: Optional covariance-adjusted projection.
    """

    vector: Tensor
    strength: float
    scope: str
    prefix_length: int
    audio_span: tuple[int, int]
    anchor: Tensor | None = None
    read: Tensor | None = None

    def apply(self, hidden: Tensor) -> Tensor:
        """Change selected prefix states only.

        Args:
            hidden: Batch, sequence, width states.

        Returns:
            Corrected states, or the original tensor at zero strength.
        """
        if self.strength == 0:
            return hidden
        mask = prefix_mask(hidden.shape[1], self.prefix_length, self.audio_span, self.scope).to(hidden.device)
        values = hidden[:, mask].float()
        vector = self.vector.to(hidden.device).float()
        if self.read is not None:
            if self.anchor is None:
                msg = "Transport requires an anchor"
                raise ValueError(msg)
            response = (values - self.anchor.to(hidden.device)) @ self.read.to(hidden.device)
            delta = response.unsqueeze(-1) * vector
        else:
            delta = vector
        changed = hidden.clone()
        changed[:, mask] = (values + self.strength * delta).to(hidden.dtype)
        return changed


@contextmanager
def steer(block: nn.Module, intervention: Intervention) -> Iterator[None]:
    """Install a hook and remove it even when inference raises.

    Args:
        block: Transformer block.
        intervention: Position-aware correction.

    Yields:
        Control with an active hook.
    """

    def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
        """Preserve auxiliary decoder outputs.

        Args:
            _module: Hooked block.
            _inputs: Original arguments.
            output: Tensor or tuple.

        Returns:
            Corrected hidden states and unchanged auxiliary outputs.
        """
        if isinstance(output, tuple):
            return (intervention.apply(output[0]), *output[1:])
        return intervention.apply(output)

    handle = block.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()
