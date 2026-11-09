"""ASVspoof 5 equal error rate and minimum architecture-agnostic DCF."""

import numpy as np
from sklearn.metrics import roc_curve

C_MISS = 1.0
C_FALSE_ALARM = 10.0
SPOOF_PRIOR = 0.05


def _bonafide_error_rates(
    bonafide_scores: np.ndarray,
    spoof_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """False-alarm and miss rates for a bonafide-positive score.

    Args:
        bonafide_scores: Scores for bonafide trials. Higher means more bonafide.
        spoof_scores: Scores for spoof trials, on the same scale.

    Returns:
        False-alarm rate (spoof accepted as bonafide) and miss rate (bonafide rejected as spoof)
        at each ROC threshold.

    """
    labels = np.concatenate([np.ones(bonafide_scores.shape[0]), np.zeros(spoof_scores.shape[0])])
    scores = np.concatenate([bonafide_scores, spoof_scores])
    false_alarm, true_positive, _thresholds = roc_curve(labels, scores, pos_label=1)
    rates: tuple[np.ndarray, np.ndarray] = (
        np.asarray(false_alarm, dtype=np.float64),
        np.asarray(np.subtract(1.0, true_positive), dtype=np.float64),
    )
    return rates


def equal_error_rate(bonafide_scores: np.ndarray, spoof_scores: np.ndarray) -> float:
    """Equal error rate from bonafide-positive scores.

    Args:
        bonafide_scores: Scores for bonafide trials. Higher means more bonafide.
        spoof_scores: Scores for spoof trials, on the same scale.

    Returns:
        Mean of the false-alarm and miss rates at the threshold where they are closest.

    """
    false_alarm, miss = _bonafide_error_rates(bonafide_scores, spoof_scores)
    index = int(np.argmin(np.abs(false_alarm - miss)))
    return float((false_alarm[index] + miss[index]) / 2.0)


def min_a_dcf(
    bonafide_scores: np.ndarray,
    spoof_scores: np.ndarray,
    *,
    cost_miss: float = C_MISS,
    cost_false_alarm: float = C_FALSE_ALARM,
    spoof_prior: float = SPOOF_PRIOR,
) -> float:
    """Normalized minimum architecture-agnostic detection cost.

    A miss is a bonafide trial scored as spoof. A false alarm is a spoof trial scored as bonafide.
    The cost is divided by `min(cost_miss * (1 - spoof_prior), cost_false_alarm * spoof_prior)`.

    Args:
        bonafide_scores: Scores for bonafide trials. Higher means more bonafide.
        spoof_scores: Scores for spoof trials, on the same scale.
        cost_miss: Cost of rejecting a bonafide trial.
        cost_false_alarm: Cost of accepting a spoof trial.
        spoof_prior: Prior probability of a spoof trial.

    Returns:
        Minimum normalized a-DCF over ROC thresholds.

    """
    false_alarm, miss = _bonafide_error_rates(bonafide_scores, spoof_scores)
    costs = cost_miss * (1.0 - spoof_prior) * miss + cost_false_alarm * spoof_prior * false_alarm
    default_cost = min(cost_miss * (1.0 - spoof_prior), cost_false_alarm * spoof_prior)
    return float(np.min(costs) / default_cost)
