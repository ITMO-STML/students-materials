"""Bounded supervised LoRA fitting smoke, with exact baseline restoration."""

import time
from typing import Any

import numpy as np
import torch
from torch import Tensor

from src.alm_steering.runtime import FrozenALM


def fitting_smoke(model: FrozenALM, audio: Tensor, labels: list[int]) -> dict[str, Any]:
    """Run two supervised updates on eight support clips, then restore weights.

    Only the checkpoint's existing q/v LoRA parameters are updated. Whisper,
    Q-Former, projection and base LLM weights remain frozen. No reasoning targets
    are synthesized. This is a fitting test, not the final fine-tuning comparator.

    Args:
        model: Loaded ALM in evaluation mode.
        audio: Eight balanced cached support recordings.
        labels: Zero for bonafide and one for spoof.

    Returns:
        Losses, update magnitude, runtime and restoration verification.
    """
    before = model.score(audio[:4])
    adapters = {name: value for name, value in model.model.named_parameters() if "lora_" in name}
    snapshot = {name: value.detach().clone() for name, value in adapters.items()}
    for parameter in adapters.values():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(adapters.values(), lr=1e-5, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")
    losses = []
    started = time.perf_counter()
    try:
        model.llm.train()
        for _step in range(2):
            optimizer.zero_grad(set_to_none=True)
            total = 0.0
            for start in range(0, len(audio), 2):
                answers = [
                    f"Final Answer: {'bonafide' if label == 0 else 'spoof'}</s>" for label in labels[start : start + 2]
                ]
                targets = model.tokenizer(answers, padding=True, add_special_tokens=False, return_tensors="pt")
                ids = targets.input_ids.to(model.device)
                mask = targets.attention_mask.to(model.device)
                with model.model.maybe_autocast():
                    prefix, _ = model.prefix(audio[start : start + 2])
                    inputs = torch.cat([prefix, model.embedding(ids)], dim=1)
                    attention = torch.cat([torch.ones(prefix.shape[:2], device=model.device), mask], dim=1)
                    logits = model.llm(inputs_embeds=inputs, attention_mask=attention, use_cache=False).logits
                    expected = ids.masked_fill(mask == 0, -100)
                    loss = torch.nn.functional.cross_entropy(
                        logits[:, prefix.shape[1] - 1 : -1].float().reshape(-1, logits.shape[-1]),
                        expected.reshape(-1),
                    )
                    scaled = loss / (len(audio) / 2)
                if not torch.isfinite(loss):
                    message = "Nonfinite ALM supervised loss"
                    raise ValueError(message)
                scaler.scale(scaled).backward()
                total += float(scaled.detach())
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(list(adapters.values()), 1.0)
            if not torch.isfinite(norm):
                message = "Nonfinite LoRA gradients"
                raise ValueError(message)
            scaler.step(optimizer)
            scaler.update()
            losses.append(total)
        change = max(float((value.detach() - snapshot[name]).abs().max()) for name, value in adapters.items())
        assert change > 0
    finally:
        with torch.no_grad():
            for name, parameter in adapters.items():
                parameter.copy_(snapshot[name])
                parameter.requires_grad_(False)
                parameter.grad = None
        model.model.eval()
    after = model.score(audio[:4])
    np.testing.assert_array_equal(before, after)
    return {
        "status": "passed",
        "updates": 2,
        "examples": len(audio),
        "losses": losses,
        "updated_parameters": sum(value.numel() for value in adapters.values()),
        "max_update": change,
        "wall_seconds": time.perf_counter() - started,
        "baseline_restore_max_difference": float(np.max(np.abs(before - after))),
        "full_finetuning_status": "not_started_requires_confirmation_after_smoke",
    }
