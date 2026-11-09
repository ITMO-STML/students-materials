"""ASVspoof 5 waveforms and Wav2Vec2 collation."""

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import override

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.utils.data import Dataset
from transformers import Wav2Vec2FeatureExtractor

SAMPLE_RATE = 16000
NUM_SAMPLES = 64600
LABEL_TO_INDEX = {"bonafide": 0, "spoof": 1}
type WaveformExample = tuple[NDArray[np.float32], int]

log = logging.getLogger(__name__)


def parse_trial(fields: Sequence[str], *, track_eval: bool) -> tuple[str, int]:
    """Read an utterance id and binary label from one metadata row.

    Args:
        fields: Whitespace-separated metadata columns.
        track_eval: Whether the row is an ASVspoof 5 eval track-1 record. Those rows end in `-`,
            so the label is the second-to-last field. Train and dev labels are the last field.

    Returns:
        Utterance id and class index (`0` bonafide, `1` spoof).

    """
    utterance_id = fields[1]
    label_name = fields[-2] if track_eval else fields[-1]
    return utterance_id, LABEL_TO_INDEX[label_name]


def read_metadata(path: Path, *, track_eval: bool, limit: int | None) -> list[tuple[str, int]]:
    """Load utterance ids and labels from an ASVspoof 5 metadata file.

    Args:
        path: Metadata text file.
        track_eval: Use the eval-track column layout.
        limit: Maximum number of trials. `None` reads the whole file.

    Returns:
        Utterance id and class index pairs, in file order.

    """
    trials: list[tuple[str, int]] = []
    with path.open() as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            trials.append(parse_trial(stripped.split(), track_eval=track_eval))
            if limit is not None and len(trials) >= limit:
                break
    return trials


def fit_length(waveform: Tensor, num_samples: int, *, random_crop: bool) -> Tensor:
    """Crop or pad a mono waveform to a fixed length.

    Args:
        waveform: Mono audio of shape `(time,)`.
        num_samples: Target length.
        random_crop: Draw a random window when the waveform is longer than `num_samples`. Otherwise
            keep the center window.

    Returns:
        Waveform of shape `(num_samples,)`.

    """
    length = int(waveform.shape[-1])
    if length == num_samples:
        return waveform
    if length > num_samples:
        if random_crop:
            start = int(torch.randint(0, length - num_samples + 1, ()).item())
        else:
            start = (length - num_samples) // 2
        return waveform[start : start + num_samples]
    return F.pad(waveform, (0, num_samples - length))


def mean_center(waveform: Tensor) -> Tensor:
    """Subtract the per-clip mean.

    Args:
        waveform: Mono audio.

    Returns:
        Mean-centered audio of the same shape.

    """
    return waveform - waveform.mean()


def prepare_waveform(
    waveform: Tensor,
    num_samples: int,
    *,
    random_crop: bool,
    augmentation: nn.Module | None,
) -> Tensor:
    """Fit length, optionally augment, fit again, and mean-center.

    Args:
        waveform: Mono audio of shape `(time,)`.
        num_samples: Target length.
        random_crop: Random crop on the first fit. The fit after augmentation is centered.
        augmentation: Training compose. Skipped when `None`.

    Returns:
        Mean-centered waveform of shape `(num_samples,)`.

    """
    fitted = fit_length(waveform, num_samples, random_crop=random_crop)
    if augmentation is not None:
        augmented = augmentation(fitted.view(1, 1, -1), sample_rate=SAMPLE_RATE)
        if not isinstance(augmented, Tensor):
            augmented = augmented.samples
        fitted = fit_length(augmented.reshape(-1), num_samples, random_crop=False)
    return mean_center(fitted)


def _load_mono(path: Path) -> Tensor:
    waveform, sample_rate = torchaudio.load(str(path))
    waveform = waveform.mean(dim=0) if waveform.shape[0] > 1 else waveform.squeeze(0)
    if sample_rate != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sample_rate, SAMPLE_RATE)
    return waveform


class AudioDataset(Dataset[WaveformExample]):
    """ASVspoof 5 clips cropped or padded to a fixed length.

    Args:
        metadata_path: Train, dev, or eval metadata file.
        audio_dir: Directory of `{utterance_id}.flac` files.
        num_samples: Crop or pad length.
        random_crop: Random window for training. Dev and eval use the center window.
        augmentation: Training augmentation compose. Dev and eval pass `None`.
        track_eval: Eval track-1 metadata layout.
        limit: Maximum number of trials. `None` uses the whole metadata file.

    """

    def __init__(
        self,
        metadata_path: Path | str,
        audio_dir: Path | str,
        num_samples: int = NUM_SAMPLES,
        *,
        random_crop: bool = False,
        augmentation: nn.Module | None = None,
        track_eval: bool = False,
        limit: int | None = None,
    ) -> None:
        self.audio_dir = Path(audio_dir)
        self.num_samples = num_samples
        self.random_crop = random_crop
        self.augmentation = augmentation
        self.trials = read_metadata(Path(metadata_path), track_eval=track_eval, limit=limit)

    def __len__(self) -> int:
        """Return the number of trials."""
        return len(self.trials)

    @override
    def __getitem__(self, index: int) -> WaveformExample:
        """Load one clip and its label.

        Args:
            index: Trial index.

        Returns:
            Mean-centered waveform of shape `(num_samples,)` and a binary label. A clip that fails
            to load is silence labeled bonafide.

        """
        utterance_id, label = self.trials[index]
        path = self.audio_dir / f"{utterance_id}.flac"
        try:
            waveform = _load_mono(path)
        except (OSError, RuntimeError, TypeError, ValueError):
            log.warning("Failed to load %s; using silence labeled bonafide", path)
            silence = np.zeros(self.num_samples, dtype=np.float32)
            return silence, LABEL_TO_INDEX["bonafide"]
        prepared = prepare_waveform(
            waveform,
            self.num_samples,
            random_crop=self.random_crop,
            augmentation=self.augmentation,
        )
        array = np.asarray(prepared.detach().cpu().numpy(), dtype=np.float32)
        return array, label


def collate_audio(
    batch: Sequence[WaveformExample],
    feature_extractor: Wav2Vec2FeatureExtractor,
) -> dict[str, Tensor]:
    """Pad waveforms for Wav2Vec2.

    Mean centering is applied before collation, so the extractor must have `do_normalize` disabled.

    Args:
        batch: Waveform and label pairs.
        feature_extractor: Extractor that pads and builds the attention mask.

    Returns:
        Batch dict with `input_values`, `attention_mask`, and integer `labels`.

    """
    waveforms = [waveform for waveform, _ in batch]
    labels = torch.tensor([label for _, label in batch], dtype=torch.long)
    encoded = feature_extractor(
        waveforms,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True,
    )
    input_values: Tensor = encoded["input_values"]
    attention_mask: Tensor = encoded["attention_mask"]
    return {"input_values": input_values, "attention_mask": attention_mask, "labels": labels}
