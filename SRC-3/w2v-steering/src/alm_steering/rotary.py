"""Compatibility for persistent rotary frequencies in legacy SALMONN weights."""

from typing import override

import torch
from torch import Tensor, nn


class CheckpointRotary(nn.Module):
    """Use saved frequencies instead of the newer loader's regenerated defaults.

    Args:
        inv_freq: Original checkpoint's persistent inverse frequencies.
    """

    def __init__(self, inv_freq: Tensor) -> None:
        super().__init__()
        self.register_buffer("inv_freq", inv_freq.detach().float().clone())

    @override
    def forward(self, x: Tensor, seq_len: int) -> tuple[Tensor, Tensor]:
        """Regenerate deterministic cosine/sine tables from saved frequencies.

        Args:
            x: Attention value states supplying device and output dtype.
            seq_len: Requested rotary table length.

        Returns:
            Broadcastable cosine and sine tables.
        """
        positions = torch.arange(seq_len, device=x.device, dtype=torch.float32)
        frequencies = torch.outer(positions, self.inv_freq.float())
        phases = torch.cat([frequencies, frequencies], dim=-1)[None, None]
        return phases.cos().to(x.dtype), phases.sin().to(x.dtype)
