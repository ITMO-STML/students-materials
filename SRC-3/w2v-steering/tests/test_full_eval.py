"""Score checkpoint integrity and lazy full-protocol indexing."""

import json
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from src.steering.full_eval import EvalAudio, final_report, load_chunks, save_chunk
from src.steering.protocol import Trial, speaker_partition


def test_chunks_round_trip_without_changing_scores(tmp_path: Path) -> None:
    """Contiguous score checkpoints restore exactly, retaining all methods.

    Args:
        tmp_path: Isolated test artifact directory.
    """
    expected = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    assert save_chunk(tmp_path, 0, {"baseline": expected[:2], "sae": expected[:2] + 0.1}) == 2
    save_chunk(tmp_path, 2, {"baseline": expected[2:], "sae": expected[2:] + 0.1})
    destinations = {name: np.full(4, np.nan, dtype=np.float32) for name in ("baseline", "sae")}
    assert load_chunks(tmp_path, destinations) == 4
    np.testing.assert_array_equal(destinations["baseline"], expected)
    np.testing.assert_array_equal(destinations["sae"], expected + 0.1)


def test_chunks_reject_gaps_and_nonfinite_scores(tmp_path: Path) -> None:
    """Never silently treat incomplete or invalid checkpoints as full results.

    Args:
        tmp_path: Isolated test artifact directory.
    """
    with pytest.raises(ValueError, match="finite"):
        save_chunk(tmp_path, 0, {"baseline": np.array([np.nan])})
    save_chunk(tmp_path, 1, {"baseline": np.array([0.5])})
    with pytest.raises(ValueError, match="noncontiguous"):
        load_chunks(tmp_path, {"baseline": np.full(2, np.nan)})


def test_lazy_dataset_retains_protocol_rows_and_codecs() -> None:
    """Full evaluation must not deduplicate codec variants or sample attacks."""
    trials = [Trial("s", f"u{i}", "A21", 1, f"C{i}", "same_source", "/unused") for i in range(3)]
    dataset = EvalAudio(trials, 64600)
    assert len(dataset) == 3
    assert len(EvalAudio(trials, 64600, start=1)) == 2


def test_reports_use_all_bonafide_for_every_attack(tmp_path: Path) -> None:
    """Use the full bona group for each attack and separately filter held-out speakers.

    Args:
        tmp_path: Isolated reference and report directory.
    """
    speakers = [f"s{i}" for i in range(100)]
    chosen = [speaker for speaker in speakers if speaker_partition(speaker, 42) == "test"][:2]
    chosen += [speaker for speaker in speakers if speaker_partition(speaker, 42) == "support"][:1]
    trials = [
        Trial(speaker, f"{speaker}_{index}", attack, int(attack != "bonafide"), "C01", "shared_source", "/unused")
        for speaker in chosen
        for index, attack in enumerate(("bonafide", "bonafide", "A21", "A21", "A22"))
    ]
    (tmp_path / "selection.json").write_text(json.dumps({"threshold": 0.5}))
    cfg = OmegaConf.create({"seed": 42, "steering": {"reference_run": str(tmp_path)}})
    scores = {"baseline": np.array([0.8 if trial.label == 0 else 0.2 for trial in trials])}
    final_report(trials, scores, cfg, tmp_path)
    results = json.loads((tmp_path / "report.json").read_text())["results"]
    for attack in ("A21", "A22", "pooled"):
        assert results["full_eval"]["baseline"][attack]["n_bonafide"] == 6
        assert results["heldout_speakers"]["baseline"][attack]["n_bonafide"] == 4
    assert results["full_eval"]["baseline"]["A21"]["n_spoof"] == 6
    assert results["full_eval"]["baseline"]["A22"]["n_spoof"] == 3
    assert results["full_eval"]["baseline"]["pooled"]["n_spoof"] == 9
