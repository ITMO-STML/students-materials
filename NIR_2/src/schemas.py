"""Датаклассы для обмена данными между этапами пайплайны."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class QueryDoc:
    """Входная пара: запрос + положительный документ."""
    qid: str
    query: str
    d_plus: str
    meta: dict = field(default_factory=dict)


@dataclass
class Fact:
    """Извлечённый из d⁺ атомарный факт."""
    text: str
    type: str
    criticality: int  # 1-5

    @classmethod
    def from_dict(cls, d: dict) -> "Fact":
        return cls(
            text=str(d.get("text", "")).strip(),
            type=str(d.get("type", "")).strip(),
            criticality=int(d.get("criticality", 3)),
        )


@dataclass
class NegativeRecord:
    """Сгенерированный негатив со всеми диагностиками."""
    qid: str
    query: str
    d_plus: str
    d_minus: str
    method: str                       # "counterfactual"
    fact_text: str                    # какой факт мутировали
    fact_type: str
    fact_criticality: int
    delta: float                      # ансамблевое значение
    delta_per_encoder: dict[str, float]
    bm25_overlap: float
    ce_score: float                   # cross-encoder relevance(q, d⁻)
    llm_judge: str                    # "yes" | "no"
    difficulty_bucket: Optional[str]  # "easy" | "medium" | "hard"
    passed_filters: bool
    rejection_reason: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PositiveRecord:
    """Сгенерированный позитивный вариант d⁺ᵢ."""
    qid: str
    query: str
    d_plus: str
    d_plus_i: str
    method: str                       # "gradient"
    target_level: str                 # "near" | "mid" | "far"
    similarity_to_d_plus: float       # ансамблевое значение
    sim_per_encoder: dict[str, float]
    ce_score: float                   # CE(q, dᵢ) — должен оставаться высоким
    llm_judge: str
    passed_filters: bool
    rejection_reason: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)
