"""Per-attack reporting using the repository's unchanged detection metrics."""

import numpy as np
from sklearn.metrics import roc_curve

from src.metrics.detection import C_FALSE_ALARM, C_MISS, SPOOF_PRIOR, equal_error_rate, min_a_dcf
from src.steering.protocol import Trial


def dev_threshold(trials: list[Trial], scores: np.ndarray) -> float:
    """Choose a finite minimum-cost threshold from baseline dev scores.

    Args:
        trials: Development records with both classes.
        scores: Bonafide-positive probabilities.

    Returns:
        Threshold to freeze across interventions and evaluation attacks.
    """
    labels = np.array([t.label == 0 for t in trials])
    false_alarm, true_positive, thresholds = roc_curve(labels, scores, drop_intermediate=False)
    costs = np.add(
        np.multiply(C_MISS * (1 - SPOOF_PRIOR), np.subtract(1.0, true_positive)),
        np.multiply(C_FALSE_ALARM * SPOOF_PRIOR, false_alarm),
    )
    # Keep the reject-all operating point while making it JSON-serializable.
    thresholds = np.where(np.isfinite(thresholds), thresholds, np.nextafter(scores.max(), np.inf))
    return float(thresholds[np.argmin(costs)])


def attack_report(trials: list[Trial], scores: np.ndarray, threshold: float) -> dict[str, dict[str, float]]:
    """Evaluate each attack against the same bonafide cohort.

    Args:
        trials: Aligned scored records.
        scores: Bonafide probabilities; higher means more bonafide.
        threshold: Fixed baseline dev threshold; accept scores >= threshold.

    Returns:
        Pooled and per-attack EER, cost, false acceptance, rejection, and counts.
    """
    if len(trials) != len(scores) or not np.isfinite(scores).all():
        message = "Scores must be finite and aligned with the manifest"
        raise ValueError(message)
    bona = scores[np.array([t.label == 0 for t in trials])]
    if not len(bona):
        message = "Per-attack evaluation requires bonafide reference recordings"
        raise ValueError(message)
    rejection = float(np.mean(bona < threshold))
    result = {}
    for attack in ["pooled", *sorted({t.attack for t in trials if t.label == 1})]:
        mask = np.array([t.label == 1 and attack in ("pooled", t.attack) for t in trials])
        spoof = scores[mask]
        if not len(spoof):
            message = "Evaluation requires spoof recordings"
            raise ValueError(message)
        acceptance = float(np.mean(spoof >= threshold))
        cost = C_MISS * (1 - SPOOF_PRIOR) * rejection + C_FALSE_ALARM * SPOOF_PRIOR * acceptance
        result[attack] = {
            "eer": equal_error_rate(bona, spoof),
            "min_dcf": min_a_dcf(bona, spoof),
            "fixed_dcf": cost / min(C_MISS * (1 - SPOOF_PRIOR), C_FALSE_ALARM * SPOOF_PRIOR),
            "spoof_acceptance": acceptance,
            "bonafide_rejection": rejection,
            "n_bonafide": float(len(bona)),
            "n_spoof": float(len(spoof)),
        }
    return result
