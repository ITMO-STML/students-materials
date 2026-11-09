"""Regularized rank-one MidSteer-inspired attack representation transport."""

import torch
from torch import Tensor, nn


class RankOneSteering(nn.Module):
    """Store an input-dependent correction without a dense inference matrix.

    Args:
        anchor: Fixed point of the affine map.
        read: Covariance-adjusted source detector with read @ source = 1.
        write: Target minus source mean.
    """

    def __init__(self, anchor: Tensor, read: Tensor, write: Tensor) -> None:
        super().__init__()
        self.register_buffer("anchor", anchor)
        self.register_buffer("read", read)
        self.register_buffer("write", write)

    def steer(self, hidden: Tensor, strength: float, *, gated: bool = False) -> Tensor:
        """Apply affine transport, optionally clipping negative source responses.

        Args:
            hidden: Token activations, with hidden width on the last axis.
            strength: Interpolation strength; zero returns the original tensor.
            gated: Clip negative projections, as in the upstream controller.

        Returns:
            Corrected states. Gating is nonlinear and has no affine guarantee.
        """
        if strength == 0:
            return hidden
        response = (hidden.float() - self.anchor) @ self.read
        if gated:
            response = response.clamp_min(0)
        return hidden + (strength * response.unsqueeze(-1) * self.write).to(hidden.dtype)


def fit_transport(
    groups: tuple[Tensor, Tensor, Tensor],
    ridge: float,
    *,
    anchored: bool,
) -> RankOneSteering:
    """Fit a minimum-disturbance mean transport under regularized second moments.

    The three groups receive equal prior weight despite differing sample sizes.
    For affine mode the anchor is their mixture mean, and the second moment is
    its covariance. Equal priors make mean directions proportional to indicator
    cross-covariances. This is the rank-one, positive-definite form of MidSteer
    Eq. 19 with a ridge. Anchored mode instead fixes the bonafide mean and uses
    the second moment around that point; this is a constrained extension.
    At strength one the source mean maps exactly to the target mean. This does
    not imply matching distributions or preserving all bonafide predictions.

    Args:
        groups: Bonafide, known training spoof, and A21 support token matrices.
        ridge: Positive diagonal regularizer relative to mean diagonal moment.
        anchored: Use the bonafide mean instead of the global mixture mean.

    Returns:
        Float32 rank-one adapter on the input device; no optimizer is used.
    """
    if ridge <= 0 or any(x.ndim != 2 or len(x) < 2 for x in groups):
        message = "Require positive ridge and three nonempty token matrices"
        raise ValueError(message)
    if len({(x.shape[1], x.device) for x in groups}) != 1:
        message = "Group widths and devices must agree"
        raise ValueError(message)
    with torch.no_grad():
        means = [x.double().mean(dim=0) for x in groups]
        anchor = means[0] if anchored else torch.stack(means).mean(dim=0)
        moment = torch.zeros((len(anchor), len(anchor)), dtype=torch.float64, device=anchor.device)
        for group in groups:
            centered = group.double() - anchor
            moment.add_(centered.T @ centered, alpha=1 / (3 * len(group)))
        regularizer = ridge * moment.diagonal().mean().clamp_min(1e-12)
        moment.diagonal().add_(regularizer)
        source = means[2] - anchor
        read = torch.linalg.solve(moment, source)
        denominator = source @ read
        if not torch.isfinite(denominator) or denominator <= 1e-12:
            message = "Degenerate or nonfinite source direction"
            raise ValueError(message)
        return RankOneSteering(anchor.float(), (read / denominator).float(), (means[1] - means[2]).float())
