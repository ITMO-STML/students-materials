"""Checks for ALM classification decisions, counts, and selection criteria."""

from types import SimpleNamespace

import numpy as np
import pytest

from src.alm_steering.classification import (
    binary_metrics,
    classification_rank,
    classification_report,
    retention_allowed,
)


def test_classification_counts_and_ties() -> None:
    """Treat spoof as positive and equal likelihoods as bonafide."""
    result = binary_metrics(np.array([0, 0, 1, 1]), np.array([0.0, -1.0, -2.0, 3.0]))
    assert all(result[key] == 1 for key in ("tp", "tn", "fp", "fn"))
    assert all(result[key] == 0.5 for key in ("accuracy", "precision", "recall", "f1"))


def test_degenerate_predictions_are_explicit() -> None:
    """All-spoof predictions cannot masquerade as good bonafide detection."""
    labels = np.array([0, 0, 1, 1])
    spoof = binary_metrics(labels, -np.ones(4))
    assert spoof["accuracy"] == spoof["precision"] == 0.5
    assert spoof["recall"] == 1 and spoof["bonafide_recall"] == 0
    assert spoof["f1"] == pytest.approx(2 / 3)
    bona = binary_metrics(labels, np.ones(4))
    assert bona["precision"] == bona["recall"] == bona["f1"] == 0
    with pytest.raises(ValueError, match="finite"):
        binary_metrics(labels, np.array([0, 1, 2, np.nan]))


def test_attack_cohorts_share_all_bonafide() -> None:
    """Each attack uses all bonafide examples; pooled metrics use all scores."""
    trials = [
        SimpleNamespace(label=label, attack=attack)
        for label, attack in [(0, "bonafide"), (0, "bonafide"), (1, "A21"), (1, "A22")]
    ]
    reports = classification_report(trials, np.array([1, -1, -1, 1]))
    assert reports["A21"]["n_bonafide"] == reports["A22"]["n_bonafide"] == 2
    assert reports["A21"]["accuracy"] == pytest.approx(2 / 3)
    assert reports["A22"]["accuracy"] == pytest.approx(1 / 3)
    assert reports["pooled"]["accuracy"] == 0.5


def test_selection_prefers_f1_and_checks_retention() -> None:
    """Selection does not use rank-only detection metrics or test labels."""
    base = {"f1": 0.8, "accuracy": 0.8, "precision": 0.8, "recall": 0.8, "bonafide_recall": 0.8}
    better = {**base, "f1": 0.9}
    assert classification_rank(better) < classification_rank(base)
    assert retention_allowed(better, base)
    assert not retention_allowed({**better, "bonafide_recall": 0.7}, base)
