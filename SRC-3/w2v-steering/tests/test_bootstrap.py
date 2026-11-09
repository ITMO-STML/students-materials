"""Speaker-cluster integrity, pairing, and unchanged-metric checks."""

import numpy as np
import pytest

from src.metrics.detection import equal_error_rate
from src.steering.bootstrap import cluster_indices, speaker_bootstrap
from src.steering.protocol import Trial


def records() -> list[Trial]:
    """Build three speakers, each with bonafide and two different attacks.

    Returns:
        Nine speaker-linked test records.
    """
    return [
        Trial(speaker, f"{speaker}_{attack}", attack, int(attack != "bonafide"), "-", "-", "/unused")
        for speaker in ("s1", "s2", "s3")
        for attack in ("bonafide", "A21", "A22")
    ]


def test_clusters_retain_all_records_and_multiplicity() -> None:
    """Drawing a speaker twice repeats all their records twice, without shuffling labels."""
    groups = [np.array([0, 3]), np.array([1, 2, 4])]
    np.testing.assert_array_equal(cluster_indices(groups, np.array([1, 1, 0])), [1, 2, 4, 1, 2, 4, 0, 3])


def test_pairing_reproducibility_and_unchanged_eer() -> None:
    """Identical methods have zero paired deltas, and each draw matches the core metric."""
    trials = records()
    scores = np.array([0.9, 0.1, 0.8, 0.4, 0.6, 0.3, 0.7, 0.8, 0.2])
    report, estimates, draws = speaker_bootstrap(trials, {"baseline": scores, "copy": scores.copy()}, 20, 42)
    again, repeated, repeated_draws = speaker_bootstrap(trials, {"baseline": scores, "copy": scores.copy()}, 20, 42)
    assert report == again
    np.testing.assert_array_equal(estimates, repeated)
    np.testing.assert_array_equal(draws, repeated_draws)
    np.testing.assert_array_equal(estimates[:, 0], estimates[:, 1])
    delta = report["results"]["copy"]["A21"]["paired_delta_eer_pp"]["baseline"]
    assert delta["mean"] == delta["std"] == 0
    groups = [np.arange(index * 3, index * 3 + 3) for index in range(3)]
    for replicate, draw in enumerate(draws):
        indices = cluster_indices(groups, draw)
        bona = [index for index in indices if trials[index].label == 0]
        spoof = [index for index in indices if trials[index].attack == "A21"]
        assert estimates[replicate, 0, 1] == 100 * equal_error_rate(scores[bona], scores[spoof])
    summary = report["results"]["baseline"]["A21"]["bootstrap_eer_percent"]
    assert summary["mean"] == estimates[:, 0, 1].mean()
    assert summary["std"] == estimates[:, 0, 1].std(ddof=1)
    assert summary["std"] > 0


def test_perfect_scores_and_invalid_inputs() -> None:
    """Perfect separation stays zero; missing classes and misaligned scores fail."""
    trials = records()
    perfect = np.array([1 - trial.label for trial in trials], dtype=float)
    _, estimates, _ = speaker_bootstrap(trials, {"baseline": perfect}, 10, 1)
    assert not estimates.any()
    with pytest.raises(ValueError, match="aligned"):
        speaker_bootstrap(trials, {"baseline": perfect[:-1]}, 10, 1)
    with pytest.raises(ValueError, match="at least two replicates"):
        speaker_bootstrap(trials, {"baseline": perfect}, 1, 1)
    bona = [trial for trial in trials if trial.label == 0]
    with pytest.raises(ValueError, match="bonafide and spoof"):
        speaker_bootstrap(bona, {"baseline": np.ones(len(bona))}, 10, 1)


def test_missing_class_draws_are_rejected_jointly_and_counted() -> None:
    """A speaker draw lacking either class must not enter any method's summary."""
    trials = [records()[0], records()[4]]
    scores = {"baseline": np.array([0.9, 0.1]), "copy": np.array([0.9, 0.1])}
    report, estimates, draws = speaker_bootstrap(trials, scores, 20, 42)
    assert report["rejected_missing_class_draws"] > 0
    assert all(set(draw) == {0, 1} for draw in draws)
    assert estimates.shape == (20, 2, 2)
    assert not estimates.any()
