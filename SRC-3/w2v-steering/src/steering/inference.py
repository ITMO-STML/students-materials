"""Strict waveform loading and temporary transformer-block interventions."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio
from torch import Tensor, nn

from src.data.dataset import SAMPLE_RATE, prepare_waveform
from src.steering.protocol import Trial


def load_audio(trials: list[Trial], num_samples: int) -> Tensor:
    """Load a fixed manifest with baseline preprocessing and fail on bad audio.

    Args:
        trials: Ordered recordings to read.
        num_samples: Baseline crop/pad length.

    Returns:
        CPU waveforms, center-cropped, zero-padded, and mean-centered.
    """
    waves = []
    for trial in trials:
        waveform, rate = torchaudio.load(str(Path(trial.audio_dir) / f"{trial.utterance}.flac"))
        waveform = waveform.mean(dim=0)
        if rate != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, rate, SAMPLE_RATE)
        if waveform.numel() == 0 or not torch.isfinite(waveform).all():
            message = f"Empty or nonfinite recording: {trial.utterance}"
            raise ValueError(message)
        waves.append(prepare_waveform(waveform, num_samples, random_crop=False, augmentation=None))
    return torch.stack(waves)


def run_model(
    model: nn.Module,
    waves: Tensor,
    block: nn.Module,
    batch_size: int,
    transform: Callable[[Tensor], Tensor] | None = None,
    *,
    capture: bool = False,
) -> tuple[np.ndarray, Tensor | None]:
    """Score waveforms under a temporary, label-independent block-output hook.

    Args:
        model: Frozen detector in eval mode.
        waves: CPU waveforms.
        block: Selected transformer block.
        batch_size: Inference batch size.
        transform: Optional token-state intervention.
        capture: Retain original token activations on CPU.

    Returns:
        Bonafide probabilities and optional original hidden states.
    """
    if model.training or batch_size < 1:
        message = "Inference requires eval mode and positive batch size"
        raise ValueError(message)
    states = []

    def hook(
        _module: nn.Module,
        _args: tuple[Any, ...],
        output: Tensor | tuple[Tensor, ...],
    ) -> Tensor | tuple[Tensor, ...]:
        hidden = output if isinstance(output, Tensor) else output[0]
        if capture:
            states.append(hidden.detach().float().cpu())
        changed = hidden if transform is None else transform(hidden)
        return changed if isinstance(output, Tensor) else (changed, *output[1:])

    handle = block.register_forward_hook(hook)
    scores = []
    device = next(model.parameters()).device
    try:
        with torch.inference_mode():
            for batch in waves.split(batch_size):
                inputs = batch.to(device)
                output = model(input_values=inputs, attention_mask=torch.ones_like(inputs, dtype=torch.long))
                scores.append(output["logits"].float().softmax(dim=-1)[:, 0].cpu())
    finally:
        handle.remove()
    return torch.cat(scores).numpy(), torch.cat(states) if capture else None
