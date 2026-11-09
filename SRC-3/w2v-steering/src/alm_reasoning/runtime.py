"""Frozen reasoning SALMONN with upstream preprocessing and cached generation."""

import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import yaml
from torch import Tensor
from transformers import WhisperFeatureExtractor

from src.alm_steering.runtime import load_external


class ReasoningALM:
    """Restore the full reasoning checkpoint without legacy adapter overrides.

    Args:
        cfg: Local paths and inference configuration.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg["device"])
        torch.cuda.set_device(self.device)
        original = yaml.safe_load(Path(cfg["model_config"]).read_text())["Model"]
        upstream = load_external(Path(cfg["source_package"]))
        allowed = inspect.signature(upstream.SALMONN.__init__).parameters
        kwargs = {k: v for k, v in original["additional_kwargs"]["salmon"].items() if k in allowed}
        kwargs.update(
            llama_path=cfg["llama_path"],
            whisper_path=cfg["whisper_path"],
            lora=original["lora"]["enabled"],
            lora_rank=original["lora"]["rank"],
            lora_alpha=original["lora"]["alpha"],
            lora_dropout=original["lora"]["lora_dropout"],
            max_txt_len=original["max_txt_len"],
            prompt_path="",
            multi_prompt=False,
            use_class_weights=False,
            torch_dtype=torch.bfloat16,
        )
        self.model = upstream.SALMONN(**kwargs)
        checkpoint = torch.load(cfg["checkpoint"], weights_only=True, mmap=True, map_location="cpu")
        if not all(key.startswith("model.") for key in checkpoint["model"]):
            message = "Unexpected checkpoint namespace"
            raise ValueError(message)
        state = {key.removeprefix("model."): value for key, value in checkpoint["model"].items()}
        self.model.load_state_dict(state, strict=True, assign=True)
        self.model.requires_grad_(False).to(self.device).eval()
        self.model.maybe_autocast = lambda dtype=None: torch.autocast("cuda", dtype=dtype or torch.bfloat16)
        self.llm = self.model.llama_model
        self.embedding = self.llm.get_input_embeddings()
        self.tokenizer = self.model.llama_tokenizer
        self.processor = WhisperFeatureExtractor.from_pretrained(cfg["whisper_path"], local_files_only=True)
        lora = self.llm.peft_config["default"]
        self.audit = {
            "strict_load": True,
            "loaded_keys": len(state),
            "epoch": checkpoint["epoch"],
            "lora_rank": lora.r,
            "lora_alpha": lora.lora_alpha,
            "use_rslora": lora.use_rslora,
            "target_modules": sorted(lora.target_modules),
            "all_parameters_frozen": not any(p.requires_grad for p in self.model.parameters()),
            "generation": checkpoint["config"].get("generation"),
            "prefix": "BOS + audio, no text prompt (upstream evaluation default)",
            "precision": "Checkpoint mixed BF16 backbone / FP32 adapters; BF16 autocast",
        }
        spec = importlib.util.spec_from_file_location("alm_reasoning_dataset", cfg["dataset_source"])
        if spec is None or spec.loader is None:
            message = "Cannot import upstream audio preprocessing"
            raise ImportError(message)
        dataset = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = dataset
        spec.loader.exec_module(dataset)
        self.augmenter = dataset.AudioAugmenter(
            target_sr=16000,
            max_len_sec=8.0,
            min_len_sec=1.0,
            pad_mode="repeat",
            crop_mode="center",
            vad_cfg=dataset.VadConfig(),
            aug_cfg=dataset.AugConfig(enabled=False),
        )
        # Fail visibly if VAD is unavailable instead of silently using upstream's exception fallback.
        dataset.torchaudio.functional.vad(torch.zeros(1, 16000), sample_rate=16000)
        self.closing = {
            tuple(self.tokenizer.encode(text, add_special_tokens=False)[-3:])
            for text in ("</answer>", "<answer>Real</answer>")
        }

    def encode(self, paths: list[Path]) -> Tensor:
        """Apply upstream evaluation audio processing, then encode audio only.

        Args:
            paths: Audio paths, without labels or reference reasoning.

        Returns:
            GPU audio-prefix embeddings.
        """
        waves = []
        for path in paths:
            wave, rate = sf.read(path, dtype="float32", always_2d=True)
            if not wave.size or not np.isfinite(wave).all():
                message = f"Invalid waveform: {path.name}"
                raise ValueError(message)
            processed, _ = self.augmenter(torch.from_numpy(wave.T.copy()), rate)
            waves.append(processed.squeeze(0).numpy())
        features = self.processor(waves, sampling_rate=16000, return_tensors="pt")["input_features"]
        with torch.inference_mode(), self.model.maybe_autocast():
            audio, _ = self.model.encode_speech(features.to(self.device))
        if not torch.isfinite(audio).all():
            message = "Nonfinite audio embeddings"
            raise ValueError(message)
        return audio

    def generate(self, audio: Tensor, max_tokens: int, *, use_cache: bool = True) -> list[dict[str, Any]]:
        """Greedily decode until an answer closes, EOS, or the fixed token budget.

        Args:
            audio: Encoded recordings with identical prefix lengths.
            max_tokens: Prespecified generation cap, independent of labels.
            use_cache: Disable only for compatibility verification.

        Returns:
            Raw continuations, token IDs, and explicit termination reasons.
        """
        sequences: list[list[int]] = [[] for _ in range(len(audio))]
        reasons = ["token_limit"] * len(audio)
        done = [False] * len(audio)
        with torch.inference_mode(), self.model.maybe_autocast():
            bos = torch.full((len(audio), 1), self.tokenizer.bos_token_id, dtype=torch.long, device=self.device)
            prefix = torch.cat((self.embedding(bos), audio), dim=1)
            current = prefix
            past = None
            for _ in range(max_tokens):
                output = self.llm(inputs_embeds=current, past_key_values=past, use_cache=use_cache, return_dict=True)
                logits = output.logits[:, -1]
                if not torch.isfinite(logits).all():
                    message = "Nonfinite generation logits"
                    raise ValueError(message)
                next_ids = logits.argmax(dim=-1)
                for index, token in enumerate(next_ids.tolist()):
                    if done[index]:
                        continue
                    sequences[index].append(token)
                    if token == self.tokenizer.eos_token_id:
                        reasons[index], done[index] = "eos", True
                    elif tuple(sequences[index][-3:]) in self.closing:
                        reasons[index], done[index] = "answer_closed", True
                if all(done):
                    break
                if use_cache:
                    past = output.past_key_values
                    if past is None:
                        message = "Upstream model did not return a generation cache"
                        raise ValueError(message)
                    current = self.embedding(next_ids[:, None])
                else:
                    prefix = torch.cat((prefix, self.embedding(next_ids[:, None])), dim=1)
                    current = prefix
        texts = self.tokenizer.batch_decode(sequences, skip_special_tokens=True)
        return [
            {"text": text, "token_ids": tokens, "generated_tokens": len(tokens), "finish_reason": reason}
            for text, tokens, reason in zip(texts, sequences, reasons, strict=True)
        ]
