"""External labels, cohort weighting, preprocessing, and resume safety."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torchaudio

from src.data.dataset import prepare_waveform
from src.steering.external_eval import external_results, validate_identity
from src.steering.external_protocol import ExternalAudio, ExternalRecording, inventory, read_elevenlabs, read_wildspoof


def test_wildspoof_collapses_trials_with_consistent_binary_labels(tmp_path: Path) -> None:
    """Target and nontarget are bonafide; repeats preserve multiplicity.

    Args:
        tmp_path: Temporary dataset root.
    """
    protocol = tmp_path / "protocol_test_label.tsv"
    protocol.write_text("s1 u1 target spoofceleb\ns2 u1 nontarget spoofceleb\ns3 u2 spoof asvspoof5\n")
    rows = read_wildspoof(tmp_path)
    assert [(r.key, r.label, r.multiplicity) for r in rows] == [("u1", 0, 2), ("u2", 1, 1)]
    assert rows[0].speaker == "unknown"
    protocol.write_text("s1 u1 target x\ns2 u1 spoof x\n")
    with pytest.raises(ValueError, match="Conflicting"):
        read_wildspoof(tmp_path)
    protocol.write_text("s1 u1 fake x\n")
    with pytest.raises(ValueError, match="Invalid"):
        read_wildspoof(tmp_path)


def test_elevenlabs_only_prepared_segments_and_explicit_generators(tmp_path: Path) -> None:
    """Avoid original/segment duplication and retain generator metadata.

    Args:
        tmp_path: Temporary dataset root.
    """
    for name in ("s/bonafide/RU/a_1.wav", "s/spoof/elevenlabs/RU/b_1.wav", "s/spoof/finevoice/EN/c_2.wav"):
        path = tmp_path / "processed_audio" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"placeholder")
    (tmp_path / "original.wav").write_bytes(b"excluded")
    rows = read_elevenlabs(tmp_path)
    assert len(rows) == 3
    assert {r.group for r in rows} == {"bonafide", "elevenlabs", "finevoice"}
    assert rows[0].source == "s/bonafide/RU/a"
    assert inventory(rows)["n_bonafide"] == 1
    with pytest.raises(ValueError, match="unique"):
        inventory(rows + rows[:1])
    Path(rows[0].path).unlink()
    with pytest.raises(FileNotFoundError):
        inventory(rows)


def test_audio_matches_original_center_crop_preprocessing(tmp_path: Path) -> None:
    """WAV input receives identical mono, center crop and mean subtraction.

    Args:
        tmp_path: Temporary audio directory.
    """
    path = tmp_path / "clip.wav"
    signal = torch.stack([torch.linspace(-0.5, 0.5, 100), torch.linspace(0.2, -0.2, 100)])
    torchaudio.save(str(path), signal, 16000)
    decoded, _ = torchaudio.load(str(path))
    row = ExternalRecording("clip", str(path), 0, "bonafide")
    index, wave = ExternalAudio([row], 64)[0]
    assert index == 0
    torch.testing.assert_close(wave, prepare_waveform(decoded.mean(0), 64, random_crop=False, augmentation=None))


def test_external_cohorts_keep_bonafide_and_explicit_trial_weights() -> None:
    """Unique-query and enrollment-weighted detection have different denominators."""
    rows = [
        ExternalRecording("b1", "unused", 0, "spoofceleb", multiplicity=3),
        ExternalRecording("b2", "unused", 0, "asvspoof5"),
        ExternalRecording("s1", "unused", 1, "spoofceleb"),
        ExternalRecording("s2", "unused", 1, "asvspoof5"),
        ExternalRecording("s3", "unused", 1, "tts-track"),
    ]
    scores = {"baseline": np.array([0.9, 0.2, 0.1, 0.8, 0.3])}
    results = external_results(rows, scores, 0.5, "wildspoof_2026")
    unique = results["all_unique_queries"]["metrics"]["baseline"]
    weighted = results["protocol_trial_weighted"]["metrics"]["baseline"]
    assert unique["pooled"]["classification"]["accuracy"] == pytest.approx(3 / 5)
    assert weighted["pooled"]["classification"]["accuracy"] == pytest.approx(5 / 7)
    assert unique["tts-track"]["n_bonafide"] == 2
    assert weighted["tts-track"]["n_bonafide"] == 4
    assert results["excluding_asvspoof5"]["n"] == 3
    assert results["datasource_tts-track"]["status"] == "not_estimable"
    assert unique["pooled"]["delta_eer_pp"] == 0
    with pytest.raises(ValueError, match="complete"):
        external_results(rows, {"baseline": scores["baseline"][:-1]}, 0.5, "wildspoof_2026")


def test_elevenlabs_generator_cohort_retains_shared_bonafide() -> None:
    """Generator comparisons retain the same bona reference and exclude other spoofs."""
    rows = [
        ExternalRecording(str(i), "unused", label, group)
        for i, (label, group) in enumerate(
            [(0, "bonafide"), (0, "bonafide"), (1, "elevenlabs"), (1, "finevoice")],
        )
    ]
    result = external_results(rows, {"baseline": np.array([0.8, 0.7, 0.2, 0.4])}, 0.5, "elevenlabs")
    cohort = result["elevenlabs_generator_and_all_bonafide"]
    assert cohort["n"] == 3
    assert cohort["metrics"]["baseline"]["pooled"]["n_bonafide"] == 2
    assert "finevoice" not in cohort["metrics"]["baseline"]


def test_resume_rejects_changed_inputs_and_orphan_chunks(tmp_path: Path) -> None:
    """Resuming must not mix different models or manifests.

    Args:
        tmp_path: Temporary run directory.
    """
    identity = {"manifest_sha256": "original", "batch_size": 4}
    validate_identity(tmp_path, identity)
    path = tmp_path / "provenance.json"
    path.write_text(json.dumps({"identity": identity}))
    validate_identity(tmp_path, identity)
    with pytest.raises(ValueError, match="differs"):
        validate_identity(tmp_path, {**identity, "manifest_sha256": "changed"})
    path.unlink()
    (tmp_path / "chunks").mkdir()
    (tmp_path / "chunks/000000000_000000001.npz").touch()
    with pytest.raises(ValueError, match="without provenance"):
        validate_identity(tmp_path, identity)
