"""Tests for reasoning answer parsing and full-cohort aggregation."""

import json
from pathlib import Path

import pytest

from src.alm_reasoning.metrics import Cohorts, parse_answer, summarize
from src.steering.protocol import Trial


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("<think>fake? real?</think><answer>Real</answer>", 0),
        ("<answer> fake </ANSWER>", 1),
        ("<answer>Fake", -1),
        ("Final Answer: spoof", -1),
        ("<answer>Real or Fake</answer>", -1),
        ("<answer>Real</answer><answer>Fake</answer>", -1),
        ("<answer>Real</answer><answer>", -1),
    ],
)
def test_parser(text: str, expected: int) -> None:
    """Require one unambiguous complete answer tag."""
    assert parse_answer(text) == expected


def test_invalid_predictions_count_as_errors() -> None:
    """Never improve accuracy by dropping invalid generations."""
    result = summarize([[2, 1, 1], [1, 2, 1]])
    assert result["accuracy"] == 0.5
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == 0.5
    assert result["f1"] == pytest.approx(4 / 7)
    assert result["invalid"] == 2


def test_pooled_attack_groups() -> None:
    """Attack metrics reuse all bonafide and are not batch averages."""
    cohorts = Cohorts()
    for index, (label, attack, pred) in enumerate(
        [(0, "bonafide", 0), (0, "bonafide", -1), (1, "A21", 1), (1, "A22", 0)],
    ):
        cohorts.add(Trial("speaker", str(index), attack, label, "-", "-", "/unused-audio-fixture"), pred)
    report = cohorts.report()["full_evaluation"]
    assert report["pooled"]["n"] == 4
    assert report["A21"]["n_bonafide"] == report["A22"]["n_bonafide"] == 2
    assert report["A21"]["accuracy"] == pytest.approx(2 / 3)
    assert report["A22"]["accuracy"] == pytest.approx(1 / 3)


def test_chunk_resume_rejects_misalignment(tmp_path: Path) -> None:
    """Resume verifies recording identity and re-parses saved generated answers."""
    from src.alm_reasoning.artifacts import restore_chunks

    trials = [Trial("speaker", "one", "A21", 1, "-", "-", "/unused-audio-fixture")]
    chunk = {
        "start": 0,
        "wall_seconds": 2.0,
        "predictions": [{"utterance": "one", "text": "<answer>Fake</answer>", "prediction": 1}],
    }
    path = tmp_path / "000000000.json"
    path.write_text(json.dumps(chunk))
    cohorts = Cohorts()
    assert restore_chunks(tmp_path, trials, cohorts) == (1, 2.0)
    assert cohorts.report()["full_evaluation"]["pooled"]["accuracy"] == 1
    chunk["predictions"][0]["utterance"] = "wrong"
    path.write_text(json.dumps(chunk))
    with pytest.raises(ValueError, match="provenance"):
        restore_chunks(tmp_path, trials, Cohorts())
