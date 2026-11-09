"""Paired speaker-cluster bootstrap of frozen held-out antispoofing scores."""

import argparse
import csv
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.metrics.detection import equal_error_rate
from src.steering.evaluation import attack_report
from src.steering.protocol import Trial, check_disjoint

log = logging.getLogger(__name__)


def cluster_indices(groups: list[np.ndarray], draw: np.ndarray) -> np.ndarray:
    """Expand sampled speakers to all their recordings, retaining duplicates.

    Args:
        groups: Recording indices for each speaker in sorted speaker order.
        draw: Speaker indices sampled with replacement.

    Returns:
        Recording indices with whole-speaker multiplicities preserved.
    """
    return np.concatenate([groups[int(index)] for index in draw])


def distribution_summary(values: np.ndarray) -> dict[str, float]:
    """Summarize bootstrap replicates, without confusing std with standard error.

    Args:
        values: At least two finite replicate statistics in reporting units.

    Returns:
        Arithmetic mean, sample std (ddof=1), and percentile 95% interval.
    """
    lower, upper = np.quantile(values, [0.025, 0.975])
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
        "ci95_low": float(lower),
        "ci95_high": float(upper),
    }


def validate_inputs(trials: list[Trial], scores: dict[str, np.ndarray], replicates: int) -> None:
    """Reject inputs that cannot define a paired speaker bootstrap.

    Args:
        trials: Test manifest.
        scores: Aligned per-method score vectors, including baseline.
        replicates: Requested number of resamples.

    Raises:
        ValueError: Budgets, labels, scores, or speaker counts are invalid.
    """
    if replicates < 2 or not trials or "baseline" not in scores:
        message = "Require at least two replicates, nonempty trials, and baseline scores"
        raise ValueError(message)
    if any(value.shape != (len(trials),) or not np.isfinite(value).all() for value in scores.values()):
        message = "Scores must be finite vectors aligned with the test manifest"
        raise ValueError(message)
    if len({trial.speaker for trial in trials}) < 2 or any(trial.label not in (0, 1) for trial in trials):
        message = "Require at least two speakers and binary labels"
        raise ValueError(message)


def speaker_bootstrap(
    trials: list[Trial],
    scores: dict[str, np.ndarray],
    replicates: int,
    seed: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Resample speakers jointly across methods, attacks, and bonafide trials.

    Args:
        trials: Fixed test recordings, including bonafide references.
        scores: Aligned, finite bonafide-positive scores for each frozen method.
        replicates: Number of valid speaker-bootstrap replicates (at least two).
        seed: Bootstrap-only random seed, not an adaptation seed.

    Returns:
        Summary metadata, EER-percent array shaped replicate/method/attack, and
        sampled speaker-index array. Draws missing a required class are rejected
        jointly for all methods/attacks, counted, and bounded to avoid hangs.
    """
    validate_inputs(trials, scores, replicates)
    methods = list(scores)
    speakers = sorted({trial.speaker for trial in trials})
    speaker_array = np.array([trial.speaker for trial in trials])
    groups = [np.flatnonzero(speaker_array == speaker) for speaker in speakers]
    attacks = ["pooled", *sorted({trial.attack for trial in trials if trial.label == 1})]
    bona = np.array([trial.label == 0 for trial in trials])
    spoof = np.array(
        [[trial.label == 1 and attack in ("pooled", trial.attack) for trial in trials] for attack in attacks],
    )
    if not bona.any() or not spoof.any(axis=1).all():
        message = "Every reported comparison requires bonafide and spoof recordings"
        raise ValueError(message)
    point = np.array(
        [[100 * equal_error_rate(scores[method][bona], scores[method][mask]) for mask in spoof] for method in methods],
    )
    estimates = np.empty((replicates, len(methods), len(attacks)), dtype=np.float64)
    draws = np.empty((replicates, len(speakers)), dtype=np.int32)
    rng = np.random.default_rng(seed)
    accepted, rejected = 0, 0
    while accepted < replicates:
        draw = rng.integers(len(speakers), size=len(speakers))
        indices = cluster_indices(groups, draw)
        bona_indices = indices[bona[indices]]
        spoof_indices = [indices[mask[indices]] for mask in spoof]
        if not len(bona_indices) or any(not len(indices) for indices in spoof_indices):
            rejected += 1
            if rejected > 10 * replicates:
                message = "Too many speaker draws lack a required class; bootstrap is not estimable"
                raise ValueError(message)
            continue
        for method_index, method in enumerate(methods):
            bona_scores = scores[method][bona_indices]
            for attack_index, indices in enumerate(spoof_indices):
                estimates[accepted, method_index, attack_index] = 100 * equal_error_rate(
                    bona_scores,
                    scores[method][indices],
                )
        draws[accepted] = draw
        accepted += 1
        if accepted % 100 == 0:
            log.info("Completed %s/%s paired speaker replicates", accepted, replicates)
    results: dict[str, dict[str, Any]] = {}
    for method_index, method in enumerate(methods):
        results[method] = {}
        for attack_index, attack in enumerate(attacks):
            values = estimates[:, method_index, attack_index]
            results[method][attack] = {
                "point_eer_percent": float(point[method_index, attack_index]),
                "bootstrap_eer_percent": distribution_summary(values),
                "paired_delta_eer_pp": {
                    reference: distribution_summary(values - estimates[:, methods.index(reference), attack_index])
                    for reference in ("baseline", "sae")
                    if reference in methods
                },
            }
    return (
        {
            "method": "Paired nonparametric speaker-cluster bootstrap of all test speakers with replacement",
            "replicates": replicates,
            "rejected_missing_class_draws": rejected,
            "seed": seed,
            "n_speakers": len(speakers),
            "n_recordings": len(trials),
            "std_ddof": 1,
            "methods": methods,
            "attacks": attacks,
            "speakers": speakers,
            "interpretation": (
                "Mean +/- sample std across test-speaker resamples, conditional on frozen models, support, "
                "and selections. Not training-seed variability, std across attacks, or mean +/- confidence interval. "
                "Original point estimates are retained; bootstrap means may differ. "
                "Percentile intervals and paired differences are exploratory, without multiplicity correction."
            ),
            "results": results,
        },
        estimates,
        draws,
    )


def summarize_run(directory: Path, replicates: int, seed: int) -> Path:
    """Revalidate original metrics and save bootstrap summaries and raw draws.

    Args:
        directory: Completed run containing manifest, scores, and report JSON.
        replicates: Valid replicate budget.
        seed: Resampling seed.

    Returns:
        New bootstrap JSON path; original run artifacts remain unchanged.
    """
    sources = [directory / name for name in ("manifest.json", "scores.json", "report.json")]
    raw, saved, report = [json.loads(path.read_text()) for path in sources]
    partitions = {name: [Trial(**row) for row in rows] for name, rows in raw.items()}
    check_disjoint(partitions)
    trials = partitions["test"]
    if saved["utterances"] != [trial.utterance for trial in trials]:
        message = "Saved score order does not match the test manifest"
        raise ValueError(message)
    scores = {method: np.asarray(values, dtype=float) for method, values in saved["scores"].items()}
    if set(scores) != set(report["results"]):
        message = "Report and scores contain different methods"
        raise ValueError(message)
    for method, values in scores.items():
        recalculated = attack_report(trials, values, float(report["threshold"]))
        for attack, metrics in recalculated.items():
            for metric, value in metrics.items():
                np.testing.assert_allclose(value, report["results"][method][attack][metric], atol=1e-12, rtol=0)
    log.info("Revalidated all original test metrics for %s methods on %s recordings", len(scores), len(trials))
    result, estimates, draws = speaker_bootstrap(trials, scores, replicates, seed)
    result["validation"] = "All original EER, cost, acceptance/rejection rates and counts recomputed and matched."
    result["source_sha256"] = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [*sources, Path(__file__), Path("src/metrics/detection.py"), Path("src/steering/evaluation.py")]
    }
    destination = directory / "bootstrap_mean_std.json"
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(directory / "bootstrap_replicates.npz", eer_percent=estimates, speaker_draws=draws)
    with destination.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "method",
                "attack",
                "point_eer_percent",
                "bootstrap_mean_eer_percent",
                "bootstrap_std_eer_pp",
                "ci95_low_percent",
                "ci95_high_percent",
                "n_replicates",
            ],
        )
        for method, attacks in result["results"].items():
            for attack, row in attacks.items():
                values = row["bootstrap_eer_percent"]
                writer.writerow(
                    [
                        method,
                        attack,
                        row["point_eer_percent"],
                        values["mean"],
                        values["std"],
                        values["ci95_low"],
                        values["ci95_high"],
                        replicates,
                    ],
                )
    return destination


def main() -> None:
    """Run a reproducible bootstrap from saved scores without model inference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261009)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    log.info("Saved bootstrap report: %s", summarize_run(args.run_directory, args.replicates, args.seed))


if __name__ == "__main__":
    main()
