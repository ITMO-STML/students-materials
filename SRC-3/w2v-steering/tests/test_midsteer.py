"""Closed-form transport invariants and timing contract checks."""

import pytest
import torch

from src.steering.compute import benchmark, measured
from src.steering.midsteer import RankOneSteering, fit_transport


@pytest.mark.parametrize("anchored", [False, True])
def test_transport_maps_source_mean_and_fixes_anchor(anchored: bool) -> None:
    """Unit strength transports the source mean and leaves the anchor fixed."""
    torch.manual_seed(42)
    groups = (torch.randn(80, 8), torch.randn(40, 8) + 2, torch.randn(20, 8) - 1)
    adapter = fit_transport(groups, 0.01, anchored=anchored)
    source, target = groups[2].mean(0), groups[1].mean(0)
    torch.testing.assert_close(adapter.steer(source, 1), target)
    torch.testing.assert_close(adapter.steer(adapter.anchor, 1), adapter.anchor)
    if anchored:
        torch.testing.assert_close(adapter.anchor, groups[0].mean(0))
    hidden = torch.randn(3, 4, 8)
    assert adapter.steer(hidden, 0) is hidden
    torch.testing.assert_close(adapter.steer(hidden, 1).mean((0, 1)), adapter.steer(hidden.mean((0, 1)), 1))


def test_regularized_transport_matches_whitened_midsteer() -> None:
    """Rank-one solve matches the whitening/pseudoinverse form of Eq. 19."""
    torch.manual_seed(10)
    groups = (torch.randn(17, 5).double(), torch.randn(23, 5).double() + 2, torch.randn(31, 5).double() - 1)
    anchor = torch.stack([group.mean(0) for group in groups]).mean(0)
    covariance = sum((group - anchor).T @ (group - anchor) / (3 * len(group)) for group in groups)
    covariance += torch.eye(5) * (0.01 * covariance.diagonal().mean())
    eigenvalues, vectors = torch.linalg.eigh(covariance)
    whitening = (vectors * eigenvalues.rsqrt()) @ vectors.T
    unwhitening = (vectors * eigenvalues.sqrt()) @ vectors.T
    source = (groups[2].mean(0) - anchor).unsqueeze(1)
    target = (groups[1].mean(0) - anchor).unsqueeze(1)
    delta = unwhitening @ (whitening @ (target - source)) @ torch.linalg.pinv(whitening @ source) @ whitening
    adapter = fit_transport(groups, 0.01, anchored=False)
    torch.testing.assert_close(adapter.write[:, None] @ adapter.read[None, :], delta.float())


def test_ridge_stabilizes_rank_deficient_tokens() -> None:
    """Repeated tokens with singular sample covariance produce finite maps."""
    groups = (torch.zeros(2, 16), torch.ones(2, 16), -torch.ones(2, 16))
    adapter = fit_transport(groups, 0.01, anchored=True)
    assert torch.isfinite(adapter.read).all()
    torch.testing.assert_close(adapter.steer(groups[2], 1), groups[1])
    with pytest.raises(ValueError, match="positive ridge"):
        fit_transport(groups, 0, anchored=True)
    with pytest.raises(ValueError, match="Degenerate"):
        fit_transport((groups[0], groups[0], groups[0]), 0.01, anchored=True)


def test_gating_clips_only_negative_projection() -> None:
    """The upstream-inspired gated variant is deliberately non-affine."""
    adapter = RankOneSteering(torch.zeros(2), torch.tensor([1.0, 0.0]), torch.tensor([0.0, 2.0]))
    inputs = torch.tensor([[-1.0, 0.0], [1.0, 0.0]])
    torch.testing.assert_close(adapter.steer(inputs, 1, gated=True), torch.tensor([[-1.0, 0.0], [1.0, 2.0]]))
    torch.testing.assert_close(adapter.steer(inputs, 1), torch.tensor([[-1.0, -2.0], [1.0, 2.0]]))


def test_benchmark_preserves_results_and_counts_rounds() -> None:
    """CPU timing preserves operation returns and measures each named job."""
    value, timing = measured(torch.device("cpu"), lambda: 7)
    assert value == 7 and timing["wall_seconds"] >= 0
    result = benchmark({"baseline": lambda: 1, "other": lambda: 2}, torch.device("cpu"), 4, 4.0, 3, 42)
    assert len(result["samples"]["baseline"]) == 3
    assert len(result["round_orders"]) == 3
    assert result["summary"]["baseline"]["overhead_percent"] == 0
