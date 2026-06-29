"""
Обёртка над LLM с тремя бэкендами:
- llama_cpp  (рекомендуется на Windows — грузит GGUF напрямую, без Ollama)
- ollama     (Linux/Mac или Windows с ASCII-путём)
- vllm       (Linux + GPU, максимальная скорость)
- transformers (универсальный fallback)
"""
from __future__ import annotations
import json, re
from typing import Any
from dataclasses import dataclass, field


@dataclass
class LLMConfig:
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    backend: str = "llama_cpp"
    max_new_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    # judge
    judge_mode: str = "cot"            # "cot" | "fast"
    judge_temperature: float = 0.0
    judge_max_new_tokens: int = 8       # FAST режим
    judge_cot_max_new_tokens: int = 320 # COT режим: хватает на 4 строки рассуждений
    # llama_cpp
    gguf_path: str = r"W:\models\qwen2.5-7b-q4.gguf"
    n_gpu_layers: int = -1        # -1 = все слои на GPU
    n_ctx: int = 4096
    # ollama
    ollama_model: str = "qwen2.5:7b"
    ollama_base_url: str = "http://localhost:11434"
    # transformers
    load_in_4bit: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "LLMConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__annotations__})


class BaseLLMClient:
    def generate(self, messages_batch, *, temperature=None, max_new_tokens=None):
        raise NotImplementedError


# ──────────────────────────────────────────────────────────────────────
# LlamaCppClient — грузит GGUF напрямую, работает на Windows без проблем
# ──────────────────────────────────────────────────────────────────────
class LlamaCppClient(BaseLLMClient):
    """
    Загружает GGUF-файл напрямую через llama-cpp-python.
    Установка: pip install llama-cpp-python
    С CUDA:    pip install llama-cpp-python --extra-index-url
               https://abetlen.github.io/llama-cpp-python/whl/cu121
    """
    def __init__(self, cfg: LLMConfig):
        from llama_cpp import Llama
        import os

        path = cfg.gguf_path
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"GGUF не найден: {path}\n"
                f"Скопируй блоб из W:\\ollama_home\\models\\blobs\\ "
                f"в {path}"
            )

        print(f"[llm] Загружаем GGUF: {path}")
        print(f"[llm] GPU слоёв: {'все' if cfg.n_gpu_layers == -1 else cfg.n_gpu_layers}")

        self.llm = Llama(
            model_path=path,
            n_gpu_layers=cfg.n_gpu_layers,
            n_ctx=cfg.n_ctx,
            verbose=False,
        )
        self.cfg = cfg
        print(f"[llm] ✓ Модель загружена")

    def generate(self, messages_batch, *, temperature=None, max_new_tokens=None):
        T = self.cfg.temperature if temperature is None else temperature
        M = self.cfg.max_new_tokens if max_new_tokens is None else max_new_tokens
        results = []
        for msgs in messages_batch:
            resp = self.llm.create_chat_completion(
                messages=msgs,
                max_tokens=M,
                temperature=T,
                top_p=self.cfg.top_p,
            )
            results.append(resp["choices"][0]["message"]["content"])
        return results


# ──────────────────────────────────────────────────────────────────────
# OllamaClient — через HTTP API
# ──────────────────────────────────────────────────────────────────────
class OllamaClient(BaseLLMClient):
    def __init__(self, cfg: LLMConfig):
        import urllib.request
        self.cfg = cfg
        self.base_url = cfg.ollama_base_url
        self.ollama_model = cfg.ollama_model
        try:
            resp = urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=3)
            data = json.loads(resp.read())
            models = [m["name"] for m in data.get("models", [])]
            print(f"[llm] Ollama доступен. Модели: {models}")
        except Exception as e:
            print(f"[llm] ⚠️  Ollama недоступен ({e})")

    def _chat(self, messages, temperature, max_tokens):
        import urllib.request as ur
        payload = json.dumps({
            "model": self.ollama_model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": self.cfg.top_p,
                "num_predict": max_tokens,
            },
        }).encode("utf-8")
        req = ur.Request(
            f"{self.base_url}/api/chat", data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with ur.urlopen(req, timeout=180) as r:
                return json.loads(r.read())["message"]["content"]
        except Exception as e:
            body = e.read().decode("utf-8") if hasattr(e, "read") else str(e)
            raise RuntimeError(f"Ollama error: {body}") from e

    def generate(self, messages_batch, *, temperature=None, max_new_tokens=None):
        T = self.cfg.temperature if temperature is None else temperature
        M = self.cfg.max_new_tokens if max_new_tokens is None else max_new_tokens
        return [self._chat(msgs, T, M) for msgs in messages_batch]


# ──────────────────────────────────────────────────────────────────────
# VLLMClient — Linux + GPU
# ──────────────────────────────────────────────────────────────────────
class VLLMClient(BaseLLMClient):
    def __init__(self, cfg: LLMConfig):
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        self.llm = LLM(model=cfg.model_name, trust_remote_code=True, dtype="bfloat16")
        self.SamplingParams = SamplingParams

    def generate(self, messages_batch, *, temperature=None, max_new_tokens=None):
        sp = self.SamplingParams(
            temperature=self.cfg.temperature if temperature is None else temperature,
            top_p=self.cfg.top_p,
            max_tokens=self.cfg.max_new_tokens if max_new_tokens is None else max_new_tokens,
        )
        prompts = [
            self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages_batch
        ]
        return [o.outputs[0].text for o in self.llm.generate(prompts, sp)]


# ──────────────────────────────────────────────────────────────────────
# HFClient — transformers fallback
# ──────────────────────────────────────────────────────────────────────
class HFClient(BaseLLMClient):
    def __init__(self, cfg: LLMConfig):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
        if torch.cuda.is_available():
            names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            vram  = [torch.cuda.get_device_properties(i).total_memory/1024**3
                     for i in range(torch.cuda.device_count())]
            print(f"[llm] GPU: {names}  VRAM: {[f'{v:.1f}GB' for v in vram]}")
            load_kwargs = dict(device_map="auto", trust_remote_code=True)
            if cfg.load_in_4bit:
                print("[llm] 4-bit quantization (bitsandbytes)")
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            else:
                load_kwargs["dtype"] = torch.bfloat16
        else:
            print("[llm] ⚠️  CPU режим")
            load_kwargs = dict(device_map="cpu", trust_remote_code=True, dtype=torch.float32)
        self.model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **load_kwargs)
        self.model.eval()
        self.torch = torch
        print(f"[llm] ✓ {cfg.model_name}")

    def generate(self, messages_batch, *, temperature=None, max_new_tokens=None):
        T = self.cfg.temperature if temperature is None else temperature
        M = self.cfg.max_new_tokens if max_new_tokens is None else max_new_tokens
        results = []
        for msgs in messages_batch:
            prompt = self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=2048
            ).to(self.model.device)
            with self.torch.no_grad():
                out = self.model.generate(
                    **inputs, max_new_tokens=M, do_sample=(T > 0),
                    temperature=max(T, 1e-6), top_p=self.cfg.top_p,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            results.append(self.tokenizer.decode(
                out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
            ))
        return results


# ──────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────
def make_client(cfg: LLMConfig) -> BaseLLMClient:
    if cfg.backend == "llama_cpp":
        return LlamaCppClient(cfg)
    elif cfg.backend == "ollama":
        return OllamaClient(cfg)
    elif cfg.backend == "vllm":
        try:
            return VLLMClient(cfg)
        except ImportError:
            print("[llm] vllm не установлен, fallback → transformers")
            return HFClient(cfg)
    elif cfg.backend == "transformers":
        return HFClient(cfg)
    else:
        raise ValueError(
            f"Unknown backend: {cfg.backend!r}. "
            f"Доступные: llama_cpp, ollama, transformers, vllm"
        )


# ──────────────────────────────────────────────────────────────────────
# Утилиты парсинга
# ──────────────────────────────────────────────────────────────────────
_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)

def extract_json(text: str) -> Any:
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = _JSON_BLOCK_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    for s_ch, e_ch in [("[", "]"), ("{", "}")]:
        s, e = text.find(s_ch), text.rfind(e_ch)
        if s >= 0 and e > s:
            try:
                return json.loads(text[s:e+1])
            except json.JSONDecodeError:
                continue
    return None

def parse_yes_no(text: str) -> str:
    if not text:
        return "unknown"
    t = text.strip().lower()
    if t.startswith("да"):  return "yes"
    if t.startswith("нет"): return "no"
    if t.startswith("yes"): return "yes"
    if t.startswith("no"):  return "no"
    return "unknown"


# Регулярки для извлечения финального вердикта из CoT-выхода.
# Идём от самой строгой формы к свободной — берём первое совпадение.
_VERDICT_PATTERNS = [
    re.compile(r"ВЕРДИКТ\s*[:\-—]?\s*\*{0,2}\s*(Да|Нет)", re.IGNORECASE),
    re.compile(r"ОТВЕТ\s*[:\-—]?\s*\*{0,2}\s*(Да|Нет)",   re.IGNORECASE),
    re.compile(r"ИТОГ\s*[:\-—]?\s*\*{0,2}\s*(Да|Нет)",    re.IGNORECASE),
    # Английский fallback, на случай если модель сорвалась в English
    re.compile(r"VERDICT\s*[:\-—]?\s*\*{0,2}\s*(Yes|No)",  re.IGNORECASE),
    re.compile(r"ANSWER\s*[:\-—]?\s*\*{0,2}\s*(Yes|No)",   re.IGNORECASE),
]


def parse_cot_verdict(text: str) -> tuple[str, str]:
    """
    Парсит выход CoT-judge: возвращает (verdict, reasoning).
    verdict ∈ {"yes", "no", "unknown"}, reasoning — текст до маркера ВЕРДИКТ.

    Стратегия:
      1. Ищем явный маркер (ВЕРДИКТ/ОТВЕТ/ИТОГ/VERDICT/ANSWER).
      2. Fallback: последняя непустая строка начинается с Да/Нет/Yes/No.
    """
    if not text:
        return "unknown", ""

    for pat in _VERDICT_PATTERNS:
        m = pat.search(text)
        if m:
            v = m.group(1).lower()
            verdict = "yes" if v in ("да", "yes") else "no"
            reasoning = text[: m.start()].strip()
            return verdict, reasoning

    # Fallback: последняя строка
    lines = [ln.strip() for ln in text.strip().split("\n") if ln.strip()]
    if lines:
        last = lines[-1].lstrip("*-—–:.> ").lower()
        # "да." / "нет," / "да, потому что" → ловим первое слово
        first_word = re.split(r"[\s,.;:!?]", last, maxsplit=1)[0]
        if first_word in ("да", "yes"):
            return "yes", "\n".join(lines[:-1])
        if first_word in ("нет", "no"):
            return "no", "\n".join(lines[:-1])

    return "unknown", text
