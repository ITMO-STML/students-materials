"""Per-token TopK SAE and residual-preserving steering."""

from typing import override

import torch
from torch import Tensor, nn


class TopKSAE(nn.Module):
    """Learn a sparse dictionary of normalized hidden states.

    Args:
        width: Input hidden width.
        expansion: Dictionary expansion factor.
        k: Maximum active features per token.
    """

    def __init__(self, width: int, expansion: int, k: int) -> None:
        super().__init__()
        if not 1 <= k <= width * expansion:
            message = "k must be between one and the dictionary size"
            raise ValueError(message)
        self.k = k
        self.register_buffer("center", torch.zeros(width))
        self.register_buffer("scale", torch.ones(()))
        self.encoder = nn.Linear(width, width * expansion)
        self.decoder = nn.Linear(width * expansion, width, bias=False)
        self.normalize_decoder()
        with torch.no_grad():
            self.encoder.weight.copy_(self.decoder.weight.T)
            self.encoder.bias.zero_()

    def normalize_decoder(self) -> None:
        """Constrain dictionary columns to unit norm."""
        with torch.no_grad():
            self.decoder.weight.div_(self.decoder.weight.norm(dim=0, keepdim=True).clamp_min(1e-8))

    def encode(self, hidden: Tensor) -> Tensor:
        """Encode tokens with nonnegative TopK sparsity.

        Args:
            hidden: States with hidden width on the last axis.

        Returns:
            Sparse features with at most k nonzeros per token.
        """
        values = self.encoder((hidden.float() - self.center) / self.scale).relu()
        top, indices = values.topk(self.k, dim=-1)
        return torch.zeros_like(values).scatter(-1, indices, top)

    @override
    def forward(self, hidden: Tensor) -> Tensor:
        """Reconstruct states in original units.

        Args:
            hidden: Input token states.

        Returns:
            Reconstructed states.
        """
        return self.decoder(self.encode(hidden)) * self.scale + self.center

    def steer(self, hidden: Tensor, direction: Tensor, strength: float) -> Tensor:
        """Multiply selected features, preserving the reconstruction residual.

        Args:
            hidden: Original states.
            direction: Sparse feature signs, positive toward spoof.
            strength: Log multiplicative strength; zero is exactly identity.

        Returns:
            h + scale * W_dec(z * (exp(strength * direction) - 1)).
        """
        if strength == 0:
            return hidden
        delta = self.encode(hidden) * torch.expm1(strength * direction)
        return hidden + (self.decoder(delta) * self.scale).to(hidden.dtype)


def active_mean(latent: Tensor) -> Tensor:
    """Pool only active temporal positions per feature.

    Args:
        latent: Features shaped (batch, time, dictionary).

    Returns:
        Pooled features, zero for inactive dimensions.
    """
    return latent.sum(dim=1) / (latent > 0).sum(dim=1).clamp_min(1)


def fit_sae(sae: TopKSAE, tokens: Tensor, steps: int, batch_size: int, seed: int) -> list[float]:
    """Fit the SAE on cached training tokens only.

    Args:
        sae: Dictionary on the target compute device.
        tokens: CPU training tokens.
        steps: Optimizer step budget.
        batch_size: Token minibatch size.
        seed: Sampling seed.

    Returns:
        Per-step normalized reconstruction losses.
    """
    if steps < 1 or batch_size < 1 or len(tokens) < 2:
        message = "SAE fitting requires positive steps/batch size and at least two tokens"
        raise ValueError(message)
    device = sae.center.device
    with torch.no_grad():
        center = tokens.mean(dim=0)
        sae.center.copy_(center.to(device))
        sae.scale.copy_((tokens - center).square().mean().sqrt().clamp_min(1e-6).to(device))
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(sae.parameters(), lr=1e-3)
    losses = []
    sae.train()
    for _ in range(steps):
        indices = torch.randint(len(tokens), (batch_size,), generator=generator)
        batch = tokens[indices].to(device)
        loss = ((sae(batch) - batch) / sae.scale).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sae.normalize_decoder()
        losses.append(float(loss.detach()))
    sae.eval().requires_grad_(False)
    return losses
