"""EER and minimum a-DCF on synthetic scores."""

import numpy as np

from src.metrics.detection import equal_error_rate, min_a_dcf


def test_separated_scores_have_zero_error() -> None:
    """Perfectly separated bonafide and spoof scores have no detection cost."""
    bonafide = np.array([0.9, 0.8, 0.95])
    spoof = np.array([0.1, 0.2, 0.05])
    assert equal_error_rate(bonafide, spoof) == 0.0
    assert min_a_dcf(bonafide, spoof) == 0.0


def test_tied_scores_match_the_default_cost() -> None:
    """Identical scores cannot beat the default detection cost."""
    scores = np.array([0.5, 0.5, 0.5, 0.5])
    assert equal_error_rate(scores, scores) == 0.5
    assert min_a_dcf(scores, scores) == 1.0
