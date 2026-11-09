"""Tests for native SASV classification, split isolation and cached steering."""

import pytest
import torch
from torch import nn

from src.alm_sasv.metrics import classification, parse_answer, report
from src.alm_sasv.protocol import choose, pair_partition
from src.alm_sasv.steering import allowed, prefill_steer
from src.alm_steering.interventions import Intervention
from src.steering.protocol import speaker_partition


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("<think>yes no gen</think><answer>gen</answer>", 2),
        ("<answer> YES </ANSWER>", 0),
        ("<answer>no</answer>", 1),
        ("gen", -1),
        ("<answer>Fake</answer>", -1),
        ("<answer>gen", -1),
        ("<answer>gen</answer><answer>yes</answer>", -1),
    ],
)
def test_strict_native_parser(text: str, expected: int) -> None:
    """Never infer a native class from words in the reasoning body."""
    assert parse_answer(text) == expected


def test_binary_collapse_and_invalid_accounting() -> None:
    """A speaker-verification error can still be correct binary antispoofing."""
    result = classification([0, 1, 2, 2], [1, 1, 2, -1])
    assert result["accuracy"] == 0.5
    assert result["binary"]["accuracy"] == 0.75
    assert result["binary"]["precision"] == 1
    assert result["binary"]["recall"] == 0.5
    assert result["binary"]["f1"] == pytest.approx(2 / 3)
    assert result["invalid"] == result["binary"]["invalid"] == 1


def test_per_attack_uses_all_bonafide_pairs() -> None:
    """Preserve both target and nontarget bonafide trials per attack."""
    rows = [
        {"label": label, "attack": attack}
        for label, attack in [(0, "bonafide"), (1, "bonafide"), (2, "A21"), (2, "A31")]
    ]
    result = report(rows, [0, 2, 2, 0])
    assert result["by_attack"]["A21"]["n_bonafide"] == 2
    assert result["by_attack"]["A31"]["n_bonafide"] == 2


def test_two_sided_partition() -> None:
    """Both enrollment and query speakers must belong to the same pool."""
    representatives = {}
    for index in range(100):
        name = f"speaker_{index}"
        representatives.setdefault(speaker_partition(name, 42), name)
    assert pair_partition(representatives["support"], representatives["support"]) == "support"
    assert pair_partition(representatives["support"], representatives["test"]) == "mixed"


def test_support_deduplicates_queries_and_excludes_transfer_attacks() -> None:
    """A22/A31 never enter support direction fitting or tuning selection."""
    rows = [
        {
            "label": label,
            "attack": "A21" if label == 2 else "bonafide",
            "key": f"{label}:{i}",
            "query_source": f"{label}:{i}",
        }
        for label in range(3)
        for i in range(8)
    ]
    rows += [{"label": 2, "attack": "A22", "key": "transfer", "query_source": "transfer"}]
    chosen = choose(rows, 4, 42)
    assert len(chosen) == 8
    assert len({row["query_source"] for row in chosen}) == 8
    assert all(row["attack"] != "A22" for row in chosen)


def test_prefill_hook_skips_decode_and_is_removed() -> None:
    """Cached token generation must not be steered or fail prefix-span checks."""
    block = nn.Identity()
    intervention = Intervention(torch.ones(3), 0.5, "audio", 5, (1, 4))
    prefill = torch.zeros(2, 5, 3)
    token = torch.zeros(2, 1, 3)
    with prefill_steer(block, intervention):
        result = block(prefill)
        assert torch.all(result[:, 1:4] == 0.5)
        assert torch.all(result[:, [0, 4]] == 0)
        assert block(token) is token
    assert block(prefill) is prefill


def test_guard_checks_both_tasks() -> None:
    """Binary gains cannot conceal a large drop in speaker classification."""
    base = {"accuracy": 0.8, "macro_f1": 0.8, "binary": {"f1": 0.8, "bonafide_recall": 0.8}}
    assert allowed(base, base)
    assert not allowed({**base, "macro_f1": 0.7}, base)
