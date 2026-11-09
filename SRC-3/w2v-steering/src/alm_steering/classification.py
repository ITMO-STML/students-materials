"""ALM classification metrics using fixed two-answer likelihood decisions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.steering.protocol import Trial


def binary_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float | int]:
    """Measure classification with spoof positive and a fixed zero boundary.

    Args:
        labels: Zero for bonafide and one for spoof.
        scores: Log P(bonafide answer) minus log P(spoof answer).

    Returns:
        Accuracy, spoof precision/recall/F1, counts and bonafide recall.
        Zero-denominator precision/recall/F1 are explicitly zero. Ties predict
        bonafide, matching argmax over the ordered bonafide/spoof answers.
    """
    labels, scores = np.asarray(labels), np.asarray(scores)
    if labels.ndim != 1 or labels.shape != scores.shape or not len(labels):
        message = "Require aligned nonempty one-dimensional labels and scores"
        raise ValueError(message)
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all():
        message = "Require binary labels and finite scores"
        raise ValueError(message)
    positive = labels == 1
    predicted = scores < 0
    tp = int(np.sum(positive & predicted))
    tn = int(np.sum(~positive & ~predicted))
    fp = int(np.sum(~positive & predicted))
    fn = int(np.sum(positive & ~predicted))
    return {
        "accuracy": (tp + tn) / len(labels),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "bonafide_recall": tn / (tn + fp) if tn + fp else 0.0,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "n_bonafide": tn + fp,
        "n_spoof": tp + fn,
    }


def classification_report(trials: Sequence[Trial], scores: np.ndarray) -> dict[str, dict[str, float | int]]:
    """Pair every attack with all bonafide records in the supplied cohort.

    Args:
        trials: Aligned cohort metadata, preserving codec variants.
        scores: Fixed-answer log likelihood differences.

    Returns:
        Pooled and per-attack classification metrics, never batch averages.
    """
    labels = np.asarray([trial.label for trial in trials])
    scores = np.asarray(scores)
    pooled = binary_metrics(labels, scores)
    if not pooled["n_bonafide"] or not pooled["n_spoof"]:
        message = "Attack reports require both bonafide and spoof examples"
        raise ValueError(message)
    reports = {"pooled": pooled}
    for attack in sorted({trial.attack for trial in trials if trial.label == 1}):
        mask = np.asarray([trial.label == 0 or trial.attack == attack for trial in trials])
        reports[attack] = binary_metrics(labels[mask], scores[mask])
    return reports


def classification_rank(metrics: dict[str, float | int]) -> tuple[float, ...]:
    """Rank tuning results by descending F1, accuracy, precision and recall.

    Args:
        metrics: Classification metrics for the A21/bonafide tuning group.

    Returns:
        Ascending-sort key, with no test information or threshold fitting.
    """
    return tuple(-float(metrics[name]) for name in ("f1", "accuracy", "precision", "recall"))


def retention_allowed(candidate: dict[str, float | int], baseline: dict[str, float | int]) -> bool:
    """Require development F1/accuracy retention and bounded bonafide loss.

    Args:
        candidate: Development pooled metrics.
        baseline: Original model development metrics.

    Returns:
        Whether F1/accuracy drop at most two points and bonafide recall five.
    """
    return (
        candidate["f1"] >= baseline["f1"] - 0.02
        and candidate["accuracy"] >= baseline["accuracy"] - 0.02
        and candidate["bonafide_recall"] >= baseline["bonafide_recall"] - 0.05
    )
