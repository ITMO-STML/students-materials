"""Read-only legacy SALMONN restoration and label-independent class scoring."""

import importlib.util
import inspect
import json
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from peft import LoraConfig
from torch import Tensor
from transformers import WhisperFeatureExtractor

from src.alm_steering.rotary import CheckpointRotary
from src.steering.protocol import Trial


def load_external(directory: Path) -> types.ModuleType:
    """Import upstream code under an isolated name, avoiding its src package.

    Args:
        directory: Read-only upstream SALMON package directory.

    Returns:
        Imported SALMONN implementation module.
    """
    name = "alm_external_salmon"
    spec = importlib.util.spec_from_file_location(
        name,
        directory / "__init__.py",
        submodule_search_locations=[str(directory)],
    )
    if spec is None or spec.loader is None:
        msg = f"Cannot import SALMONN from {directory}"
        raise ImportError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return sys.modules[f"{name}.salmonn"]


def legacy_lora(**kwargs: Any) -> LoraConfig:
    """Restore legacy q/v LoRA rather than upstream's newer all-module rsLoRA.

    Args:
        **kwargs: Arguments supplied by the current upstream loader.

    Returns:
        Original-checkpoint-compatible LoRA configuration.
    """
    kwargs.update(target_modules=["q_proj", "v_proj"], use_rslora=False)
    return LoraConfig(**kwargs)


class FrozenALM:
    """Restore the selected SALMONN adapter over local Vicuna and Whisper weights.

    Args:
        cfg: Runtime paths, inventory, precision, and prompt configuration.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg["device"])
        torch.cuda.set_device(self.device)
        self.dtype = torch.float16
        upstream = load_external(Path(cfg["source_package"]))
        upstream.LoraConfig = legacy_lora
        inventory = json.loads(Path(cfg["inventory"]).read_text())
        original = inventory["selected"]["config"]["model"]
        allowed = inspect.signature(upstream.SALMONN.__init__).parameters
        kwargs = {key: value for key, value in original.items() if key in allowed}
        kwargs.update(
            llama_path=cfg["llama_path"],
            whisper_path=cfg["whisper_path"],
            prompt_path="",
            multi_prompt=False,
            use_class_weights=False,
            torch_dtype=self.dtype,
        )
        self.model = upstream.SALMONN(**kwargs)
        checkpoint = torch.load(
            inventory["selected"]["checkpoint"],
            weights_only=True,
            map_location="cpu",
            mmap=True,
        )
        state = checkpoint["model"]
        rotary = [key for key in state if key.endswith(".rotary_emb.inv_freq")]
        for key in rotary:
            parent = self.model.get_submodule(key.removesuffix(".rotary_emb.inv_freq"))
            parent.rotary_emb = CheckpointRotary(state[key])
        weights = state
        missing, unexpected = self.model.load_state_dict(weights, strict=False)
        # Frozen backbone tensors are intentionally absent from these adapter-only checkpoints.
        bad_missing = [
            key for key in missing if not key.startswith(("speech_encoder.", "llama_model.")) or "lora_" in key
        ]
        if unexpected or bad_missing:
            msg = f"Incompatible checkpoint: unexpected={unexpected}, missing={bad_missing}"
            raise ValueError(msg)
        for name, parameter in self.model.named_parameters():
            if name in state:
                torch.testing.assert_close(parameter.detach().cpu(), state[name].to(parameter.dtype), rtol=0, atol=0)
        self.load_audit = {
            "checkpoint": inventory["selected"]["checkpoint"],
            "checkpoint_sha256": inventory["checkpoint_sha256"],
            "loaded_keys": len(weights),
            "restored_rotary_buffers": len(rotary),
            "rotary_policy": "Recompute cosine/sine from saved inverse frequencies, not newer upstream defaults",
            "expected_missing_frozen_keys": len(missing),
            "unexpected_keys": unexpected,
            "lora": "q_proj/v_proj rank8 alpha32; rsLoRA disabled",
        }
        self.model.requires_grad_(False)
        self.model.to(self.device).eval()
        # Match the historical model's inference autocast instead of the newer loader's BF16 default.
        self.model.maybe_autocast = lambda dtype=None: torch.autocast("cuda", dtype=dtype or self.dtype)
        self.tokenizer = self.model.llama_tokenizer
        self.llm = self.model.llama_model
        self.embedding = self.llm.get_input_embeddings()
        self.layers = self.llm.base_model.model.model.layers
        self.processor = WhisperFeatureExtractor.from_pretrained(cfg["whisper_path"], local_files_only=True)
        self.prompt = cfg["prompt"]

    def audio_embeddings(self, trials: list[Trial]) -> Tensor:
        """Encode exact recordings using native SALMONN audio preprocessing.

        Args:
            trials: Metadata; class labels are not used.

        Returns:
            CPU FP16 projected audio-prefix tokens.
        """
        waves = []
        for trial in trials:
            wave, rate = sf.read(Path(trial.audio_dir) / f"{trial.utterance}.flac", dtype="float32")
            if rate != 16000 or wave.ndim != 1 or not len(wave) or not np.isfinite(wave).all():
                msg = f"Invalid 16 kHz mono waveform: {trial.utterance}"
                raise ValueError(msg)
            waves.append(np.pad(wave[:480000], (0, max(0, 16000 - len(wave)))))
        features = self.processor(waves, sampling_rate=16000, return_tensors="pt")["input_features"]
        with torch.inference_mode(), self.model.maybe_autocast():
            embeddings, _ = self.model.encode_speech(features.to(self.device))
        if not torch.isfinite(embeddings).all():
            msg = "Nonfinite audio embeddings"
            raise ValueError(msg)
        return embeddings.half().cpu()

    def prefix(self, audio: Tensor, prompt: str | None = None) -> tuple[Tensor, tuple[int, int]]:
        """Construct the upstream BOS, prompt, audio, prompt ordering.

        Args:
            audio: Cached projected audio tokens.
            prompt: Optional controlled prompt replacement.

        Returns:
            Prefix embeddings and half-open audio-token span.
        """
        prompt = prompt or self.prompt
        before, after = prompt.split("<SpeechHere>")
        before_ids = self.tokenizer(before, add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)
        after_ids = self.tokenizer(after, add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)
        bos = torch.full((len(audio), 1), self.tokenizer.bos_token_id, device=self.device, dtype=torch.long)
        embeds = torch.cat(
            [
                self.embedding(bos),
                self.embedding(before_ids).expand(len(audio), -1, -1),
                audio.to(self.device, self.dtype),
                self.embedding(after_ids).expand(len(audio), -1, -1),
            ],
            dim=1,
        )
        start = 1 + before_ids.shape[1]
        return embeds, (start, start + audio.shape[1])

    def score(self, audio: Tensor, prompt: str | None = None) -> np.ndarray:
        """Compute log P(bonafide answer) minus log P(spoof answer).

        Both complete verbalizers are scored for every input. No sample label,
        ground-truth answer prefix, or ground-truth reasoning enters inference.

        Args:
            audio: Cached projected audio tokens.
            prompt: Optional controlled prompt replacement.

        Returns:
            Finite bonafide-positive log likelihood ratios.
        """
        with torch.no_grad(), self.model.maybe_autocast():
            prefix, _ = self.prefix(audio, prompt)
            likelihoods = []
            for answer in ("Final Answer: bonafide</s>", "Final Answer: spoof</s>"):
                ids = self.tokenizer(answer, add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)
                ids = ids.expand(len(audio), -1)
                inputs = torch.cat([prefix, self.embedding(ids)], dim=1)
                outputs = self.llm(inputs_embeds=inputs, use_cache=False, return_dict=True)
                logits = outputs.logits[:, prefix.shape[1] - 1 : -1].float()
                logp = logits.log_softmax(-1).gather(-1, ids.unsqueeze(-1)).squeeze(-1).sum(-1)
                likelihoods.append(logp)
            scores = (likelihoods[0] - likelihoods[1]).cpu().numpy()
        if not np.isfinite(scores).all():
            msg = "Nonfinite class scores"
            raise ValueError(msg)
        return scores

    def greedy(self, audio: Tensor, max_tokens: int = 16) -> list[str]:
        """Generate label-free diagnostic answers without upstream cache assumptions.

        Args:
            audio: Cached projected audio tokens.
            max_tokens: Hard generation limit.

        Returns:
            Greedy continuations, including visible special tokens.
        """
        with torch.inference_mode(), self.model.maybe_autocast():
            embeds, _ = self.prefix(audio)
            generated = []
            finished = torch.zeros(len(audio), device=self.device, dtype=torch.bool)
            for _ in range(max_tokens):
                logits = self.llm(inputs_embeds=embeds, use_cache=False).logits[:, -1]
                token = logits.argmax(-1)
                token = torch.where(finished, self.tokenizer.eos_token_id, token)
                generated.append(token)
                finished |= token == self.tokenizer.eos_token_id
                embeds = torch.cat([embeds, self.embedding(token[:, None])], dim=1)
                if finished.all():
                    break
        return self.tokenizer.batch_decode(torch.stack(generated, dim=1).cpu(), skip_special_tokens=False)
