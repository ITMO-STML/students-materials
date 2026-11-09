"""Text-derived directions and modality-specific activation collection."""

from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from torch import Tensor, nn

if TYPE_CHECKING:
    from src.alm_steering.runtime import FrozenALM

ANCHORS = [
    "unnatural pitch contours",
    "spectral discontinuities",
    "robotic intonation",
    "repeated acoustic patterns",
    "abrupt phonetic transitions",
    "metallic voice quality",
    "inconsistent format structure",
    "synthetic harmonics",
    "unnatural pauses",
    "overly smooth speech",
    "artificial breathing sounds",
    "digital synthesis artifacts",
    "irregular vocal timing",
    "phase discontinuities",
    "unnatural consonant boundaries",
    "mechanical rhythm",
    "inconsistent voice texture",
    "distorted sibilants",
    "vocoder artifacts",
    "repeated pitch patterns",
    "unnatural vowel duration",
    "missing microprosody",
    "discontinuous background acoustics",
    "artificial resonance",
]


def corpus() -> list[dict[str, str]]:
    """Return a fixed, evaluation-independent sentence and anchor corpus.

    Returns:
        Hand-authored descriptions of synthesis-related acoustic concepts.
    """
    return [
        {"sentence": f"The reviewer noticed {anchor} during the listening session.", "anchor": anchor}
        for anchor in ANCHORS
    ]


def anchor_positions(ids: list[int], anchor: list[int]) -> list[int]:
    """Find one exact tokenized anchor, rejecting ambiguous or missing matches.

    Args:
        ids: Full sentence token sequence.
        anchor: Independently tokenized anchor phrase.

    Returns:
        Contiguous anchor positions.
    """
    matches = [i for i in range(len(ids) - len(anchor) + 1) if ids[i : i + len(anchor)] == anchor]
    if len(matches) != 1 or not anchor:
        message = "Anchor must have one exact token match"
        raise ValueError(message)
    return list(range(matches[0], matches[0] + len(anchor)))


def text_directions(model: FrozenALM, layers: list[int], *, adapted: bool) -> dict[str, Tensor]:
    """Fit text mean-shift and PCA-probe directions without audio examples.

    Args:
        model: Frozen ALM, whose LoRA may temporarily be disabled.
        layers: One-based decoder outputs.
        adapted: Extract with existing LoRA enabled instead of the base backbone.

    Returns:
        Layer-qualified mean and probe vectors; probe norm matches mean norm.
    """
    features: dict[int, list[Tensor]] = {layer: [] for layer in layers}
    context = nullcontext() if adapted else model.llm.disable_adapter()
    with context, torch.inference_mode(), model.model.maybe_autocast():
        for row in corpus():
            ids = model.tokenizer(row["sentence"], add_special_tokens=True).input_ids
            anchor = model.tokenizer(row["anchor"], add_special_tokens=False).input_ids
            positions = anchor_positions(ids, anchor)
            controls = [i for i in range(1, len(ids)) if i not in positions]
            handles = []
            for layer in layers:

                def hook(
                    _module: nn.Module,
                    _inputs: Any,
                    output: Any,
                    layer: int = layer,
                    positions: list[int] = positions,
                    controls: list[int] = controls,
                ) -> None:
                    """Collect anchor and background means for the active sentence.

                    Args:
                        _module: Decoder block.
                        _inputs: Forward arguments.
                        output: Decoder outputs.
                        layer: Captured layer number.
                        positions: Captured anchor positions.
                        controls: Captured non-anchor positions.
                    """
                    hidden = output[0] if isinstance(output, tuple) else output
                    features[layer].append(
                        torch.stack(
                            [
                                hidden[0, positions].float().mean(0),
                                hidden[0, controls].float().mean(0),
                            ],
                        ).cpu(),
                    )

                handles.append(model.layers[layer - 1].register_forward_hook(hook))
            try:
                model.llm(input_ids=torch.tensor([ids], device=model.device), use_cache=False)
            finally:
                for handle in handles:
                    handle.remove()
    result = {}
    origin = "adapted" if adapted else "base"
    for layer in layers:
        states = torch.stack(features[layer]).numpy()
        mean = states[:, 0].mean(0) - states[:, 1].mean(0)
        x = np.concatenate([states[:, 0], states[:, 1]])
        y = np.concatenate([np.ones(len(states)), np.zeros(len(states))])
        pca = PCA(n_components=len(states) // 2, svd_solver="full")
        reduced = pca.fit_transform(x)
        probe = LogisticRegression(C=1.0, max_iter=2000, random_state=42).fit(reduced, y)
        vector = pca.components_.T @ probe.coef_[0]
        vector *= np.linalg.norm(mean) / max(np.linalg.norm(vector), 1e-12)
        result[f"text_mean_{origin}_{layer}"] = torch.from_numpy(mean.copy()).float()
        result[f"text_probe_{origin}_{layer}"] = torch.from_numpy(vector.copy()).float()
    return result


def audio_states(model: FrozenALM, audio: Tensor, layers: list[int], batch_size: int) -> dict[int, Tensor]:
    """Collect eight evenly spaced audio-prefix states per clip and layer.

    Args:
        model: Frozen ALM.
        audio: Cached projected audio tokens.
        layers: One-based decoder outputs.
        batch_size: Number of clips per forward.

    Returns:
        CPU float32 tensors shaped clips, eight positions, hidden width.
    """
    collected: dict[int, list[Tensor]] = {layer: [] for layer in layers}
    with torch.inference_mode(), model.model.maybe_autocast():
        for index in range(0, len(audio), batch_size):
            prefix, (start, end) = model.prefix(audio[index : index + batch_size])
            positions = torch.linspace(start, end - 1, 8).long()
            handles = []
            for layer in layers:

                def hook(
                    _module: nn.Module,
                    _inputs: Any,
                    output: Any,
                    layer: int = layer,
                    positions: Tensor = positions,
                ) -> None:
                    """Collect selected audio positions without altering outputs.

                    Args:
                        _module: Decoder block.
                        _inputs: Forward arguments.
                        output: Decoder outputs.
                        layer: Captured layer index.
                        positions: Captured audio positions.
                    """
                    hidden = output[0] if isinstance(output, tuple) else output
                    collected[layer].append(hidden[:, positions].float().cpu())

                handles.append(model.layers[layer - 1].register_forward_hook(hook))
            try:
                model.llm(inputs_embeds=prefix, use_cache=False)
            finally:
                for handle in handles:
                    handle.remove()
    return {layer: torch.cat(rows) for layer, rows in collected.items()}
