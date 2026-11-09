"""Focused CPU checks for ALM rotary restoration and text anchors."""

import pytest
import torch

from src.alm_steering.directions import anchor_positions, corpus
from src.alm_steering.rotary import CheckpointRotary


def test_saved_rotary_frequencies() -> None:
    """Use saved frequencies exactly, preserve state, and cast output dtype."""
    frequencies = torch.tensor([0.93, 0.071])
    rotary = CheckpointRotary(frequencies)
    frequencies.zero_()
    x = torch.zeros(1, 2, 5, 4, dtype=torch.float16)
    cosine, sine = rotary(x, 5)
    expected = torch.outer(torch.arange(5).float(), torch.tensor([0.93, 0.071]))
    expected = torch.cat([expected, expected], dim=-1)[None, None]
    torch.testing.assert_close(cosine, expected.cos().half(), rtol=0, atol=0)
    torch.testing.assert_close(sine, expected.sin().half(), rtol=0, atol=0)
    assert cosine.dtype == x.dtype
    assert "inv_freq" in rotary.state_dict()


def test_unique_anchor_positions() -> None:
    """Find exact anchors and reject absent, empty, or ambiguous phrases."""
    assert anchor_positions([1, 8, 9, 2], [8, 9]) == [1, 2]
    for ids, anchor in (([1, 2], [9]), ([1, 2], []), ([8, 9, 8, 9], [8, 9])):
        with pytest.raises(ValueError, match="one exact token match"):
            anchor_positions(ids, anchor)


def test_text_corpus_is_fixed_and_unique() -> None:
    """Keep all 24 independently authored anchors present once per sentence."""
    rows = corpus()
    assert len(rows) == 24
    assert len({row["anchor"] for row in rows}) == len(rows)
    assert all(row["sentence"].count(row["anchor"]) == 1 for row in rows)
