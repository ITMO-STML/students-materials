"""Disjoint sharding and original-format checkpoint merge invariants."""

from pathlib import Path

import numpy as np
import pytest

from src.steering.full_eval import load_chunks, save_chunk
from src.steering.parallel_eval import restore_shard, split_ranges


def test_ranges_cover_suffix_once_and_preserve_batch_boundaries() -> None:
    """Two shards must preserve all complete batches and the final partial batch."""
    ranges = split_ranges(93184, 680774, 2, 4)
    assert ranges[0][0] == 93184
    assert ranges[0][1] == ranges[1][0]
    assert ranges[-1][1] == 680774
    assert all(start % 4 == 0 for start, _ in ranges)
    assert abs((ranges[0][1] - ranges[0][0]) - (ranges[1][1] - ranges[1][0])) <= 4
    with pytest.raises(ValueError, match="batch-aligned"):
        split_ranges(3, 100, 2, 4)


def test_shards_resume_and_merge_to_original_protocol_order(tmp_path: Path) -> None:
    """Interleaved worker completion must merge identically to serial evaluation.

    Args:
        tmp_path: Isolated chunk directory.
    """
    expected = np.arange(18, dtype=np.float32) / 20
    save_chunk(tmp_path, 0, {"baseline": expected[:4]})
    save_chunk(tmp_path, 12, {"baseline": expected[12:]})
    lower, next_index = restore_shard(tmp_path, 4, 12, ["baseline"])
    assert next_index == 4 and np.isnan(lower["baseline"]).all()
    upper, next_index = restore_shard(tmp_path, 12, 18, ["baseline"])
    assert next_index == 18
    np.testing.assert_array_equal(upper["baseline"], expected[12:])
    save_chunk(tmp_path, 4, {"baseline": expected[4:12]})
    scores = {"baseline": np.full(18, np.nan)}
    assert load_chunks(tmp_path, scores) == 18
    np.testing.assert_array_equal(scores["baseline"], expected)


def test_shard_rejects_cross_boundary_chunk(tmp_path: Path) -> None:
    """Never accept a checkpoint crossing the exclusive worker ownership boundary.

    Args:
        tmp_path: Isolated chunk directory.
    """
    save_chunk(tmp_path, 4, {"baseline": np.ones(8)})
    with pytest.raises(ValueError, match="Overlapping"):
        restore_shard(tmp_path, 4, 8, ["baseline"])
