"""Deterministic checks for repeated-measurement summary statistics."""

import math

import pytest

from src.steering.compute import summarize_rounds


def test_mean_and_sample_std_use_rounds_and_paired_overhead() -> None:
    """Use sample std, transform throughput before averaging, and pair rounds."""
    samples = {
        "baseline": [{"wall_seconds": value, "extra_peak_allocated_mib": 2.0} for value in (1.0, 3.0)],
        "adapter": [{"wall_seconds": value, "extra_peak_allocated_mib": 4.0} for value in (2.0, 3.0)],
    }
    result = summarize_rounds(samples, clips=2, audio_seconds=4.0)
    base, adapter = result["baseline"], result["adapter"]
    assert base["n_rounds"] == 2
    assert base["mean_ms_per_clip"] == 1000.0
    assert base["std_ms_per_clip"] == pytest.approx(math.sqrt(500_000))
    assert base["mean_clips_per_second"] == pytest.approx(4 / 3)
    assert base["std_paired_overhead_percent"] == 0.0
    assert adapter["mean_paired_overhead_percent"] == 50.0
    assert adapter["std_paired_overhead_percent"] == pytest.approx(math.sqrt(5000))


def test_single_round_std_is_unavailable() -> None:
    """Never represent missing replication as zero variability."""
    result = summarize_rounds({"baseline": [{"wall_seconds": 1.0, "extra_peak_allocated_mib": 0.0}]}, 2, 4)
    assert result["baseline"]["std_ms_per_clip"] is None
    assert result["baseline"]["mean_ms_per_clip"] == 500.0


def test_reject_unpaired_and_invalid_rounds() -> None:
    """Require matching repetition counts and physically valid durations."""
    row = {"wall_seconds": 1.0, "extra_peak_allocated_mib": 0.0}
    with pytest.raises(ValueError, match="aligned"):
        summarize_rounds({"baseline": [row], "adapter": [row, row]}, 2, 4)
    with pytest.raises(ValueError, match="positive"):
        summarize_rounds({"baseline": [{**row, "wall_seconds": 0.0}]}, 2, 4)
