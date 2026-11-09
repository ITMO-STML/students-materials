"""ASVspoof 5 parsing, cropping, and mean centering."""

from pathlib import Path

import hydra
import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from src.data.dataset import AudioDataset, fit_length, mean_center, parse_trial, prepare_waveform

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")


def test_metadata_rows() -> None:
    """Train and eval metadata columns map to utterance ids and class indices."""
    utterance_id, label = parse_trial(["T_4850", "T_0000000000", "F", "-", "A05", "spoof"], track_eval=False)
    assert utterance_id == "T_0000000000"
    assert label == 1
    utterance_id, label = parse_trial(["T_0001", "T_0000000009", "F", "-", "-", "bonafide"], track_eval=False)
    assert utterance_id == "T_0000000009"
    assert label == 0
    utterance_id, label = parse_trial(
        ["E_2192", "E_0004993854", "F", "-", "0", "-", "-", "bonafide", "bonafide", "-"],
        track_eval=True,
    )
    assert utterance_id == "E_0004993854"
    assert label == 0
    utterance_id, label = parse_trial(
        ["E_1607", "E_0009538969", "M", "C05", "2", "E_0009486171", "AC1", "A26", "spoof", "-"],
        track_eval=True,
    )
    assert utterance_id == "E_0009538969"
    assert label == 1


def test_fit_length_and_mean_center() -> None:
    """Crops, pads, and mean centering produce a zero-mean fixed-length clip."""
    longer = torch.arange(10, dtype=torch.float32)
    center = fit_length(longer, 4, random_crop=False)
    assert center.tolist() == [3.0, 4.0, 5.0, 6.0]
    shorter = torch.ones(2)
    padded = fit_length(shorter, 4, random_crop=False)
    assert padded.tolist() == [1.0, 1.0, 0.0, 0.0]
    cropped = fit_length(longer, 4, random_crop=True)
    assert cropped.shape == (4,)
    centered = mean_center(torch.tensor([1.0, 3.0]))
    assert torch.allclose(centered, torch.tensor([-1.0, 1.0]))
    prepared = prepare_waveform(longer, 4, random_crop=False, augmentation=None)
    assert prepared.shape == (4,)
    assert torch.allclose(prepared.mean(), torch.tensor(0.0), atol=1e-6)


def test_augmentation_config() -> None:
    """The compose always runs, and each transform probability is in (0.2, 0.9)."""
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        cfg = compose(config_name="main")
    assert cfg.data.root == "/mnt/shared_ru.ml.SZ-2_000049/antispoofing_datasets/asvspoof5"
    assert cfg.data.limit is None
    assert cfg.data.augmentation.p == 1.0
    assert cfg.data.augmentation.transforms[0]._target_ == "torch_audiomentations.PitchShift"
    assert cfg.data.augmentation.transforms[0].p == 0.5
    noise = cfg.data.augmentation.transforms[1:5]
    assert [transform._target_ for transform in noise] == ["torch_audiomentations.AddColoredNoise"] * 4
    assert cfg.data.augmentation.transforms[5]._target_ == "torch_audiomentations.HighPassFilter"
    assert cfg.data.augmentation.transforms[6]._target_ == "torch_audiomentations.LowPassFilter"
    for transform in cfg.data.augmentation.transforms:
        assert 0.2 < transform.p < 0.9
    augmentation = hydra.utils.instantiate(cfg.data.augmentation)
    waveform = torch.randn(1, 1, 1600)
    augmented = augmentation(waveform, sample_rate=16000)
    assert isinstance(augmented, torch.Tensor)
    assert augmented.shape[0] == 1


def test_failed_load_returns_silence_bonafide(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A clip that cannot be decoded becomes silence labeled bonafide."""
    metadata = tmp_path / "metadata.txt"
    metadata.write_text("T_1 T_0000000001 F - A01 spoof\n")

    def fail(path: Path) -> torch.Tensor:
        message = f"broken audio: {path}"
        raise TypeError(message)

    monkeypatch.setattr("src.data.dataset._load_mono", fail)
    dataset = AudioDataset(metadata, tmp_path, num_samples=8)
    waveform, label = dataset[0]
    assert waveform.shape == (8,)
    assert waveform.dtype == np.float32
    assert np.all(waveform == 0)
    assert label == 0


def test_one_training_clip() -> None:
    """One mounted training trial loads, crops, and mean-centers."""
    root = Path("/mnt/shared_ru.ml.SZ-2_000049/antispoofing_datasets/asvspoof5")
    metadata = root / "ASVspoof5.train.metadata.txt"
    if not metadata.exists():
        pytest.skip("ASVspoof 5 is not mounted")
    dataset = AudioDataset(metadata, root / "flac_T", num_samples=64600, random_crop=False, limit=1)
    waveform, label = dataset[0]
    assert waveform.shape == (64600,)
    assert waveform.dtype == np.float32
    assert abs(float(waveform.mean())) < 1e-5
    assert label in (0, 1)
