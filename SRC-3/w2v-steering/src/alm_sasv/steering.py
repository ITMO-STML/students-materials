"""Prefill-only steering and direction fitting for native SASV reasoning."""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch
from torch import Tensor, nn

from src.alm_steering.directions import audio_states, text_directions
from src.alm_steering.interventions import Intervention
from src.steering.midsteer import fit_transport


@contextmanager
def prefill_steer(block: nn.Module, intervention: Intervention) -> Iterator[None]:
    """Intervene on prefix states, never on cached one-token reasoning steps.

    Args:
        block: Selected decoder block.
        intervention: Additive or rank-one prefix intervention.

    Yields:
        Scope with a removable inference-only hook.
    """

    def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
        """Preserve KV cache and skip generated-token-only forward calls.

        Args:
            _module: Hooked decoder.
            _inputs: Decoder inputs.
            output: Decoder hidden states and optional cached keys/values.

        Returns:
            Prefix-corrected output or the unchanged generation output.
        """
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.shape[1] < intervention.prefix_length:
            return output
        changed = intervention.apply(hidden)
        return (changed, *output[1:]) if isinstance(output, tuple) else changed

    handle = block.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def fit_directions(model: Any, audio: dict[str, Tensor], manifest: dict[str, Any], layers: list[int]) -> dict[str, Any]:
    """Fit four steering families without test labels or parameter updates.

    Args:
        model: Frozen paired SASV runtime.
        audio: Support and development-spoof projected tokens.
        manifest: Ordered support and reference metadata.
        layers: Prespecified one-based decoder layers.

    Returns:
        Audio-contrast, covariance-transport and text-derived directions.
    """
    result = {}
    states = {name: audio_states(model, audio[name], layers, model.cfg["batch_size"]) for name in ("support", "known")}
    spoof = torch.tensor([row["label"] == 2 for row in manifest["support"]])
    for layer in layers:
        bona = states["support"][layer][~spoof].flatten(0, 1)
        attack = states["support"][layer][spoof].flatten(0, 1)
        known = states["known"][layer].flatten(0, 1)
        mean = attack.mean(0) - bona.mean(0)
        result[f"audio_mean_{layer}"] = {"vector": mean}
        transport = fit_transport(
            tuple(group.to(model.device) for group in (bona, known, attack)),
            0.01,
            anchored=False,
        )
        result[f"transport_{layer}"] = {
            "vector": transport.write.cpu(),
            "read": transport.read.cpu(),
            "anchor": transport.anchor.cpu(),
        }
        generator = torch.Generator().manual_seed(42 + layer)
        random = torch.randn(mean.shape, generator=generator)
        result[f"random_{layer}"] = {"vector": random * mean.norm() / random.norm()}
    for key, vector in text_directions(model, layers, adapted=True).items():
        result[key.replace("_adapted", "")] = {"vector": vector}
    return result


def allowed(candidate: dict[str, Any], baseline: dict[str, Any]) -> bool:
    """Guard native speaker classification and binary bonafide retention.

    Args:
        candidate: Complete development metrics of a candidate.
        baseline: Frozen development baseline metrics.

    Returns:
        Whether both classification tasks stay within prespecified tolerances.
    """
    return (
        candidate["accuracy"] >= baseline["accuracy"] - 0.02
        and candidate["macro_f1"] >= baseline["macro_f1"] - 0.02
        and candidate["binary"]["f1"] >= baseline["binary"]["f1"] - 0.02
        and candidate["binary"]["bonafide_recall"] >= baseline["binary"]["bonafide_recall"] - 0.05
    )
