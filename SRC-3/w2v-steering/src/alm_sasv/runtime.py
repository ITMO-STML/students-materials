"""Strict restoration of paired SALMONN using the cleaned-up project code."""

import importlib
import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import yaml
from torch import Tensor
from transformers import WhisperFeatureExtractor

from src.alm_reasoning.runtime import ReasoningALM


def external_module(name: str, path: Path) -> types.ModuleType:
    """Import a project file without shadowing this repository's src package.

    Args:
        name: Isolated module name.
        path: Existing upstream Python file.

    Returns:
        Imported module.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        message = f"Cannot import {path}"
        raise ImportError(message)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class SASVALM(ReasoningALM):
    """Frozen paired SASV model; all wrapper tensors must match the checkpoint.

    Args:
        cfg: Experiment paths and fixed generation settings.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg["device"])
        torch.cuda.set_device(self.device)
        root = Path(cfg["source_root"])
        package = types.ModuleType("alm_clean_models")
        package.__path__ = [str(root / "src/models")]
        sys.modules[package.__name__] = package
        upstream = importlib.import_module("alm_clean_models.sasv_salmon")
        original = yaml.safe_load(Path(cfg["checkpoint_config"]).read_text())
        test = yaml.safe_load(Path(cfg["test_config"]).read_text())
        model_cfg = original["Model"]
        model_cfg["ckpt"] = ""  # Strict restricted full-state load below; never drop saved adapters.
        model_cfg["additional_kwargs"]["salmon"]["new_lora"] = False
        self.wrapper = upstream.SASVSalmonModel(model_cfg)
        checkpoint = torch.load(cfg["checkpoint"], weights_only=True, mmap=True, map_location="cpu")
        self.wrapper.load_state_dict(checkpoint["model"], strict=True, assign=True)
        self.wrapper.requires_grad_(False).to(self.device).eval()
        self.model = self.wrapper.model
        self.model.maybe_autocast = lambda dtype=None: torch.autocast("cuda", dtype=dtype or torch.bfloat16)
        self.llm = self.model.llama_model
        self.llm.gradient_checkpointing_disable()
        self.embedding = self.llm.get_input_embeddings()
        self.tokenizer = self.model.llama_tokenizer
        self.layers = self.llm.base_model.model.model.layers
        assets = model_cfg["additional_kwargs"]["salmon"]["pretrained_ckpts"]
        self.processor = WhisperFeatureExtractor.from_pretrained(assets["whisper_path"], local_files_only=True)
        self.dataset = external_module("alm_clean_dataset", root / "src/dataloaders/dataset.py")
        audio = test["Audio"]
        vad = audio.get("vad", {})
        self.augmenter = self.dataset.AudioAugmenter(
            target_sr=audio["target_sr"],
            max_len_sec=audio["test_max_len_sec"],
            min_len_sec=audio["min_len_sec"],
            pad_mode=audio["pad_mode"],
            crop_mode="center",
            vad_cfg=self.dataset.VadConfig(**vad),
            aug_cfg=self.dataset.AugConfig(enabled=False),
            enroll_max_len_sec=audio["test_enroll_max_len_sec"],
            query_max_len_sec=audio["test_query_max_len_sec"],
        )
        self.dataset.torchaudio.functional.vad(torch.zeros(1, 16000), sample_rate=16000)
        import json

        self.prompt = json.loads((root / original["Prompts"]["reasoning_prompts"]).read_text())["sasv"][0]
        self.silence = float(model_cfg["additional_kwargs"]["salmon"]["silence_delay_seconds"])
        self.closing = {
            tuple(self.tokenizer.encode(text, add_special_tokens=False)[-3:])
            for text in ("</answer>", "<answer>gen</answer>")
        }
        lora = self.llm.peft_config["default"]
        self.audit = {
            "strict_load": True,
            "loaded_keys": len(checkpoint["model"]),
            "epoch": checkpoint["epoch"],
            "lora_rank": lora.r,
            "lora_alpha": lora.lora_alpha,
            "use_rslora": lora.use_rslora,
            "all_parameters_frozen": not any(p.requires_grad for p in self.wrapper.parameters()),
            "beats_present": self.model.beats is not None,
            "wrapper_ce2_head_restored": self.wrapper.bonafide_spoof_head is not None,
            "embedded_generation": checkpoint["config"].get("generation"),
            "audio_config": {k: v for k, v in audio.items() if k != "aug"},
            "prompt": self.prompt,
            "prompt_policy": "First native reasoning template, deterministic; no ground truth",
            "overrides": "Checkpoint model and reasoning; test-config audio and dev sample cap",
        }

    def encode_pairs(self, rows: list[dict[str, Any]]) -> Tensor:
        """Encode enrollment-query pairs with upstream collation and no labels.

        Args:
            rows: Manifest entries with paths; targets are never passed to model.

        Returns:
            CPU BF16 projected acoustic tokens.
        """
        items = []
        for row in rows:
            waves = []
            for key, kind in (("reference", "enroll"), ("query", "query")):
                wave, rate = sf.read(row[key], dtype="float32", always_2d=True)
                if not wave.size or not np.isfinite(wave).all():
                    message = f"Invalid audio: {row[key]}"
                    raise ValueError(message)
                value, _ = self.augmenter(torch.from_numpy(wave.T.copy()), rate, kind)
                waves.append(value)
            items.append(
                {
                    "audio_id": row["id"],
                    "reference_audios": [waves[0]],
                    "query_audios": [waves[1]],
                    "reference_speaker_ids": [],
                    "query_speaker_ids": [],
                    "text": "",
                    "reasoning": "",
                    "gt": "",
                },
            )
        batch = self.dataset.collate_fn(
            items,
            processor=self.processor,
            model_name="sasv_salmon",
            prompt_templates=[self.prompt],
            silence_delay_seconds=self.silence,
            deterministic_prompts=True,
        )
        with torch.inference_mode(), self.model.maybe_autocast():
            speech, _ = self.model.encode_speech(
                batch["spectrogram"].to(self.device),
                batch["raw_wav"].to(self.device),
                batch["padding_mask"].to(self.device),
            )
        if not torch.isfinite(speech).all():
            message = "Nonfinite SASV audio embeddings"
            raise ValueError(message)
        return speech.cpu()

    def prefix(self, audio: Tensor) -> tuple[Tensor, tuple[int, int]]:
        """Build the native prompt-wrapped BOS and paired-audio prefix.

        Args:
            audio: Cached projected acoustic tokens.

        Returns:
            Full prefix and half-open acoustic-token span.
        """
        prompt = self.dataset._format_prompt_for_model(  # noqa: SLF001 - reuse exact upstream collation
            self.prompt,
            "sasv_salmon",
            reference_audios=[None],
            query_audios=[None],
        )
        before = prompt.split("<SpeechHere>")[0]
        start = 1 + len(self.tokenizer.encode(before, add_special_tokens=False))
        with torch.inference_mode(), self.model.maybe_autocast():
            speech = audio.to(self.device)
            wrapped, mask = self.model.prompt_wrap(
                speech,
                torch.ones(speech.shape[:2], dtype=torch.long, device=self.device),
                prompt,
                multi_prompt=False,
            )
            if not mask.all():
                message = "Unexpected padding in fixed native prefix"
                raise ValueError(message)
            bos = torch.full((len(audio), 1), self.tokenizer.bos_token_id, dtype=torch.long, device=self.device)
            prefix = torch.cat((self.embedding(bos), wrapped), dim=1)
        return prefix, (start, start + audio.shape[1])

    def decode(self, audio: Tensor, max_tokens: int | None = None, *, use_cache: bool = True) -> list[dict[str, Any]]:
        """Decode reasoning with a context-window-aware hard generation limit.

        Args:
            audio: Cached acoustic tokens.
            max_tokens: Optional engineering-smoke override.
            use_cache: Disable only for cache parity checks.

        Returns:
            Saved text, token IDs and termination reasons.
        """
        prefix, _ = self.prefix(audio)
        cap = min(max_tokens or self.cfg["max_new_tokens"], self.llm.config.max_position_embeddings - prefix.shape[1])
        if cap <= 0:
            message = "Audio/text prefix exceeds model context window"
            raise ValueError(message)
        return super().generate(prefix[:, 1:], cap, use_cache=use_cache)
