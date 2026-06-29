"""
Оркестрация двух пайплайн:

1. NegativePipeline (CN) — Counterfactual Negatives
   QueryDoc → факты → контрфакты → δ-стратификация → BM25 → CE+LLM → NegativeRecord

2. PositivePipeline (GP) — Gradient Positives
   QueryDoc → ветка из 3 версий (near/mid/far) → sim-targeting → CE+LLM → PositiveRecord

Оба возвращают итераторы записей (можно стримить в JSONL).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterator

from .schemas import QueryDoc, Fact, NegativeRecord, PositiveRecord
from .llm_client import BaseLLMClient, extract_json
from .encoders import EncoderPool
from .filters import DeltaFilter, BM25Filter, DualJudge
from .prompts import (
    build_fact_extraction_messages,
    build_counterfactual_messages,
    build_gradient_positive_messages,
)

log = logging.getLogger("pipelines")


# ──────────────────────────────────────────────────────────────────────
# Counterfactual Negatives
# ──────────────────────────────────────────────────────────────────────

@dataclass
class NegConfig:
    max_facts_per_doc: int = 5
    max_mutations_per_fact: int = 1
    min_criticality: int = 3


class NegativePipeline:
    def __init__(
        self,
        llm: BaseLLMClient,
        encoders: EncoderPool,
        delta_filter: DeltaFilter,
        bm25_filter: BM25Filter,
        judge: DualJudge,
        cfg: NegConfig,
    ):
        self.llm = llm
        self.encoders = encoders
        self.delta_filter = delta_filter
        self.bm25_filter = bm25_filter
        self.judge = judge
        self.cfg = cfg

    # --- этап 1: извлечение фактов ---
    def extract_facts(self, qd: QueryDoc) -> list[Fact]:
        msgs = build_fact_extraction_messages(qd.query, qd.d_plus)
        raw = self.llm.generate([msgs])[0]
        parsed = extract_json(raw)
        if not isinstance(parsed, list):
            log.warning("[CN] не удалось распарсить факты для qid=%s", qd.qid)
            return []
        facts = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            try:
                f = Fact.from_dict(item)
            except (ValueError, TypeError):
                continue
            # факт должен реально быть в d⁺ (LLM иногда галлюцинирует)
            if f.text and f.text in qd.d_plus and f.criticality >= self.cfg.min_criticality:
                facts.append(f)
        # сортируем по криричности убывая, берём top-K
        facts.sort(key=lambda x: -x.criticality)
        return facts[: self.cfg.max_facts_per_doc]

    # --- этап 2: контрфактическая мутация ---
    def mutate(self, qd: QueryDoc, fact: Fact) -> list[str]:
        """Генерируем max_mutations_per_fact вариантов."""
        msgs = build_counterfactual_messages(qd.query, qd.d_plus, fact.text, fact.type)
        # генерируем сразу несколько кандидатов через temperature
        batch = [msgs] * self.cfg.max_mutations_per_fact
        outs = self.llm.generate(batch)
        # минимальная санитизация: убираем кавычки-обёртки если LLM накосячила
        cleaned = []
        for o in outs:
            o = o.strip()
            if o.startswith('"') and o.endswith('"') and len(o) > 2:
                o = o[1:-1]
            cleaned.append(o)
        return cleaned

    # --- этап 3: оценка + фильтры ---
    def evaluate_candidate(
        self, qd: QueryDoc, d_minus: str, fact: Fact
    ) -> NegativeRecord:
        # δ
        delta, delta_per_enc = self.encoders.delta(qd.query, qd.d_plus, d_minus)
        bucket = self.delta_filter.assign_bucket(delta)

        # BM25 (лексическая близость к d⁺)
        bm25_ok, bm25_sim = self.bm25_filter.is_valid(qd.d_plus, d_minus)

        # CE + LLM (оба должны сказать "не релевантен")
        judge_ok, judge_info = self.judge.is_valid_negative(qd.query, d_minus)

        passed = (bucket is not None) and bm25_ok and judge_ok
        reasons = []
        if bucket is None:
            reasons.append(f"delta={delta:.3f} вне бакетов")
        if not bm25_ok:
            reasons.append(f"bm25={bm25_sim:.3f} > {self.bm25_filter.similarity_max}")
        if not judge_ok:
            reasons.append(f"judge: ce={judge_info['ce_score']:.3f}, "
                          f"llm={judge_info['llm_judge']}")

        return NegativeRecord(
            qid=qd.qid,
            query=qd.query,
            d_plus=qd.d_plus,
            d_minus=d_minus,
            method="counterfactual",
            fact_text=fact.text,
            fact_type=fact.type,
            fact_criticality=fact.criticality,
            delta=delta,
            delta_per_encoder=delta_per_enc,
            bm25_overlap=bm25_sim,
            ce_score=judge_info["ce_score"],
            llm_judge=judge_info["llm_judge"],
            difficulty_bucket=bucket,
            passed_filters=passed,
            rejection_reason="; ".join(reasons) if reasons else None,
        )

    def run_one(self, qd: QueryDoc) -> Iterator[NegativeRecord]:
        facts = self.extract_facts(qd)
        if not facts:
            log.info("[CN] qid=%s: фактов не найдено", qd.qid)
            return
        for fact in facts:
            candidates = self.mutate(qd, fact)
            for d_minus in candidates:
                if not d_minus or d_minus.strip() == qd.d_plus.strip():
                    continue  # LLM вернула оригинал
                rec = self.evaluate_candidate(qd, d_minus, fact)
                yield rec


# ──────────────────────────────────────────────────────────────────────
# Gradient Positives
# ──────────────────────────────────────────────────────────────────────

@dataclass
class PosTarget:
    name: str
    target: float
    tolerance: float


@dataclass
class PosConfig:
    branches_per_target: int = 3
    targets: list[PosTarget] = None
    must_remain_positive: bool = True


class PositivePipeline:
    def __init__(
        self,
        llm: BaseLLMClient,
        encoders: EncoderPool,
        judge: DualJudge,
        cfg: PosConfig,
    ):
        self.llm = llm
        self.encoders = encoders
        self.judge = judge
        self.cfg = cfg

    def generate_branches(self, qd: QueryDoc, n_versions: int) -> list[dict]:
        """Один батч генерации с просьбой выдать N версий разного уровня."""
        msgs = build_gradient_positive_messages(qd.query, qd.d_plus, n_versions=n_versions)
        # Запросим несколько раз с разным seed (через temperature разнообразие),
        # чтобы дерево было шире
        batch = [msgs] * self.cfg.branches_per_target
        outs = self.llm.generate(batch)
        results = []
        for raw in outs:
            parsed = extract_json(raw)
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict) and "text" in item:
                        results.append({
                            "level": str(item.get("level", "")),
                            "text": str(item["text"]).strip(),
                        })
        return results

    def assign_to_targets(
        self, qd: QueryDoc, candidates: list[dict]
    ) -> list[tuple[PosTarget, dict, float, dict]]:
        """
        Считаем sim(d⁺, dᵢ) для всех кандидатов, привязываем каждого к
        ближайшему target-уровню (если попадает в tolerance).
        """
        if not candidates:
            return []
        texts = [c["text"] for c in candidates]
        sims, per_enc = self.encoders.similarity_batch(qd.d_plus, texts, role="dd")
        attached = []
        for i, cand in enumerate(candidates):
            sim = float(sims[i])
            per_enc_i = {k: float(v[i]) for k, v in per_enc.items()}
            # найти ближайший target в пределах tolerance
            best = None
            for t in self.cfg.targets:
                if abs(sim - t.target) <= t.tolerance:
                    if best is None or abs(sim - t.target) < abs(sim - best.target):
                        best = t
            if best is not None:
                attached.append((best, cand, sim, per_enc_i))
        return attached

    def evaluate_candidate(
        self,
        qd: QueryDoc,
        cand: dict,
        target: PosTarget,
        sim: float,
        per_enc: dict,
    ) -> PositiveRecord:
        if self.cfg.must_remain_positive:
            judge_ok, judge_info = self.judge.is_valid_positive(qd.query, cand["text"])
            passed = judge_ok
            reason = (
                None if judge_ok
                else f"judge: ce={judge_info['ce_score']:.3f}, "
                     f"llm={judge_info['llm_judge']}"
            )
        else:
            judge_info = {"ce_score": -1.0, "llm_judge": "skipped"}
            passed = True
            reason = None

        return PositiveRecord(
            qid=qd.qid,
            query=qd.query,
            d_plus=qd.d_plus,
            d_plus_i=cand["text"],
            method="gradient",
            target_level=target.name,
            similarity_to_d_plus=sim,
            sim_per_encoder=per_enc,
            ce_score=judge_info["ce_score"],
            llm_judge=judge_info["llm_judge"],
            passed_filters=passed,
            rejection_reason=reason,
        )

    def run_one(self, qd: QueryDoc) -> Iterator[PositiveRecord]:
        n_levels = len(self.cfg.targets)
        candidates = self.generate_branches(qd, n_versions=n_levels)
        attached = self.assign_to_targets(qd, candidates)
        for target, cand, sim, per_enc in attached:
            yield self.evaluate_candidate(qd, cand, target, sim, per_enc)
