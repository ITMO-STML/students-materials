from typing import Union

import numpy as np
import torch
import torchaudio
from torchaudio.compliance import kaldi as torch_kaldi
from torchaudio.transforms import SlidingWindowCmn
from torchvision.transforms import Compose

from torch_audiomentations.core.transforms_interface import BaseWaveformTransform


def kaldi_compatible_fbank(
    num_mel_bins: int,
    low_freq: int,
    high_freq: int,
    sample_frequency: int,
) -> Compose:
    return Compose(
        [
            KaldiCompatibleMelSpectrogram(
                num_mel_bins=num_mel_bins,
                low_freq=low_freq,
                high_freq=high_freq,
                sample_frequency=sample_frequency,
            ),
            SlidingWindowCmn(cmn_window=300, center=True),
            add_channel,
        ]
    )


class KaldiCompatibleMelSpectrogram:
    def __init__(
        self,
        num_mel_bins: int,
        low_freq: int = 20,
        high_freq: int = 7900,
        sample_frequency: int = 16_000,
    ) -> None:
        self.num_mel_bins = num_mel_bins
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.sample_frequency = sample_frequency

    def __call__(
        self, waveform: Union[np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform)
        if waveform.dtype == torch.int16:
            waveform = waveform.float()
        return torch_kaldi.fbank(
            waveform,
            num_mel_bins=self.num_mel_bins,
            low_freq=self.low_freq,
            high_freq=self.high_freq,
            window_type="hamming",
            sample_frequency=self.sample_frequency,
            snip_edges=False,
            energy_floor=0.0,
            dither=1.0,
        )


def add_channel(fbank: torch.Tensor) -> torch.Tensor:
    return torch.unsqueeze(fbank.T, dim=0)


class ExtractFbanks64(BaseWaveformTransform):
    def __init__(self) -> None:
        super().__init__()
        self.pipeline = kaldi_compatible_fbank(
            num_mel_bins=64,
            low_freq=20,
            high_freq=3700,
            sample_frequency=8000,
        )

    def forward(
        self, wav: torch.Tensor, sample_rate: int = 8000, **kwargs
    ) -> torch.Tensor:
        if isinstance(wav, np.ndarray):
            wav = torch.from_numpy(wav).unsqueeze(0)
        if sample_rate != 8000:
            wav = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=8000
            )(wav)
        wav = self.pipeline(wav)
        return wav