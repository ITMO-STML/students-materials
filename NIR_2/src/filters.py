"""
Фильтры для отсева мусорных и false-positive/false-negative примеров.

DeltaFilter        — стратификация по бакетам сложности (вместо одного жёсткого диапазона)
BM25Filter         — отсев лексических дубликатов d⁺
CrossEncoderJudge  — релевантность(q, d) через CE
LLMJudge           — релевантность(q, d) через LLM
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .llm_client import BaseLLMClient, parse_yes_no, parse_cot_verdict
from .prompts import build_judge_messages, build_judge_cot_messages


# ──────────────────────────────────────────────────────────────────────
# DeltaFilter
# ──────────────────────────────────────────────────────────────────────

@dataclass
class Bucket:
    name: str
    min: float
    max: float
    target_per_query: int = 1


class DeltaFilter:
    """Распределяет негативы по бакетам сложности по значению δ."""

    def __init__(self, buckets: list[dict]):
        self.buckets = [Bucket(**b) for b in buckets]

    def assign_bucket(self, delta: float) -> Optional[str]:
        for b in self.buckets:
            if b.min <= delta <= b.max:
                return b.name
        return None  # вне всех бакетов — отбраковываем

    def is_valid(self, delta: float) -> bool:
        return self.assign_bucket(delta) is not None


# ──────────────────────────────────────────────────────────────────────
# BM25Filter
# ──────────────────────────────────────────────────────────────────────

class BM25Filter:
    """
    Лексическая близость d⁺ и d⁻. Если они слишком похожи лексически,
    это значит, что мутация была поверхностной → бракуем.
    """

    def __init__(self, similarity_max: float = 0.6):
        self.similarity_max = similarity_max
        # Ленивая инициализация razdel + rank_bm25
        self._tokenizer = None

    def _tokenize(self, text: str) -> list[str]:
        if self._tokenizer is None:
            try:
                from razdel import tokenize as razdel_tok
                self._tokenizer = lambda t: [x.text.lower() for x in razdel_tok(t)
                                             if x.text.isalnum()]
            except ImportError:
                # fallback на простой split
                import re
                self._tokenizer = lambda t: re.findall(r"\w+", t.lower())
        return self._tokenizer(text)

    def jaccard_similarity(self, a: str, b: str) -> float:
        """Простая метрика лексической близости. Можно заменить на BM25, но
        для пары документов Jaccard на токенах достаточен и интерпретируем."""
        ta = set(self._tokenize(a))
        tb = set(self._tokenize(b))
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)

    def is_valid(self, d_plus: str, d_minus: str) -> tuple[bool, float]:
        """True = ОК (различие достаточное)."""
        s = self.jaccard_similarity(d_plus, d_minus)
        return s <= self.similarity_max, s


# ──────────────────────────────────────────────────────────────────────
# CrossEncoderJudge
# ──────────────────────────────────────────────────────────────────────

class CrossEncoderJudge:
    """Cross-encoder релевантность для верификации."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "cuda",
        batch_size: int = 16,
        relevance_threshold: float = 0.5,
        irrelevance_threshold: float = 0.2,
    ):
        from FlagEmbedding import FlagReranker  # bge-reranker-v2-m3 поддерживает русский
        self.model = FlagReranker(model_name, use_fp16=True, devices=[device])
        self.batch_size = batch_size
        self.relevance_threshold = relevance_threshold
        self.irrelevance_threshold = irrelevance_threshold

    def score(self, query: str, doc: str) -> float:
        s = self.model.compute_score([[query, doc]], normalize=True)
        if isinstance(s, list):
            return float(s[0])
        return float(s)

    def score_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        scores = self.model.compute_score(
            [list(p) for p in pairs], normalize=True
        )
        if isinstance(scores, float):
            return [scores]
        return [float(x) for x in scores]

    def is_clearly_irrelevant(self, query: str, doc: str) -> tuple[bool, float]:
        s = self.score(query, doc)
        return s < self.irrelevance_threshold, s

    def is_clearly_relevant(self, query: str, doc: str) -> tuple[bool, float]:
        s = self.score(query, doc)
        return s >= self.relevance_threshold, s


# ──────────────────────────────────────────────────────────────────────
# LLMJudge
# ──────────────────────────────────────────────────────────────────────

class LLMJudge:
    """
    LLM-судья. Используется для второй верификации в дополнение к CE.

    Два режима:
      - mode="fast": одна строка Да/Нет (быстро, но шумит на числах/датах).
      - mode="cot":  пошаговое рассуждение + явный ВЕРДИКТ (медленнее, но
                     заметно точнее на числовых/датовых контрфактах).

    Внутренние методы возвращают только верд ("yes"/"no"/"unknown") ради совместимости
    с DualJudge.is_valid_*. Для диагностики (последний raw output и reasoning)
    предусмотрен judge_verbose(), используется в sanity-скрипте.
    """

    def __init__(
        self,
        llm: BaseLLMClient,
        mode: str = "cot",
        temperature: float = 0.0,
        max_new_tokens_fast: int = 8,
        max_new_tokens_cot: int = 320,
    ):
        if mode not in ("fast", "cot"):
            raise ValueError(f"Unknown judge mode: {mode!r}")
        self.llm = llm
        self.mode = mode
        self.temperature = temperature
        self.max_new_tokens_fast = max_new_tokens_fast
        self.max_new_tokens_cot = max_new_tokens_cot

    # --- внутренние утилиты ---

    def _build_msgs(self, query: str, doc: str) -> list[dict]:
        if self.mode == "cot":
            return build_judge_cot_messages(query, doc)
        return build_judge_messages(query, doc)

    def _max_tokens(self) -> int:
        return self.max_new_tokens_cot if self.mode == "cot" else self.max_new_tokens_fast

    def _parse_one(self, raw: str) -> tuple[str, str]:
        """Возвращает (verdict, reasoning)."""
        if self.mode == "cot":
            return parse_cot_verdict(raw)
        return parse_yes_no(raw), ""

    # --- публичный API ---

    def judge(self, query: str, doc: str) -> str:
        """Возвращает 'yes' | 'no' | 'unknown'."""
        verdict, _ = self.judge_verbose(query, doc)
        return verdict

    def judge_verbose(self, query: str, doc: str) -> tuple[str, str]:
        """Возвращает (verdict, reasoning) — нужно для отладки/отчёта."""
        msgs = self._build_msgs(query, doc)
        raw = self.llm.generate(
            [msgs],
            temperature=self.temperature,
            max_new_tokens=self._max_tokens(),
        )[0]
        return self._parse_one(raw)

    def judge_batch(self, pairs: list[tuple[str, str]]) -> list[str]:
        verdicts, _ = self.judge_batch_verbose(pairs)
        return verdicts

    def judge_batch_verbose(
        self, pairs: list[tuple[str, str]]
    ) -> tuple[list[str], list[str]]:
        """Возвращает (verdicts, reasonings) — параллельные списки."""
        msgs_batch = [self._build_msgs(q, d) for q, d in pairs]
        outs = self.llm.generate(
            msgs_batch,
            temperature=self.temperature,
            max_new_tokens=self._max_tokens(),
        )
        verdicts, reasonings = [], []
        for raw in outs:
            v, r = self._parse_one(raw)
            verdicts.append(v)
            reasonings.append(r)
        return verdicts, reasonings


# ──────────────────────────────────────────────────────────────────────
# Композитный фильтр: оба судьи должны согласиться
# ──────────────────────────────────────────────────────────────────────

class DualJudge:
    """
    CE + LLM должны быть согласны.
    Для негатива: ОБА говорят "не отвечает".
    Для позитива: ОБА говорят "отвечает".
    """

    def __init__(self, ce: CrossEncoderJudge, llm_j: LLMJudge):
        self.ce = ce
        self.llm = llm_j

    def is_valid_negative(
        self, query: str, doc: str
    ) -> tuple[bool, dict]:
        ce_irrel, ce_score = self.ce.is_clearly_irrelevant(query, doc)
        llm_verdict = self.llm.judge(query, doc)  # хотим "no"
        passed = ce_irrel and llm_verdict == "no"
        return passed, {"ce_score": ce_score, "llm_judge": llm_verdict}

    def is_valid_positive(
        self, query: str, doc: str
    ) -> tuple[bool, dict]:
        ce_rel, ce_score = self.ce.is_clearly_relevant(query, doc)
        llm_verdict = self.llm.judge(query, doc)  # хотим "yes"
        passed = ce_rel and llm_verdict == "yes"
        return passed, {"ce_score": ce_score, "llm_judge": llm_verdict}
