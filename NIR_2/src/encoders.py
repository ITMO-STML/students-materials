"""
Ансамбль энкодеров для вычисления δ-метрики (семантической близости).

Принцип: используем несколько независимых энкодеров (e5, bge-m3) и берём
взвешенное среднее косинусных сходств. Это снижает selection bias по
сравнению с использованием одного энкодера, который потом будет дообучаться.

API:
    pool = EncoderPool.from_config(cfg)
    sim, per_enc = pool.similarity(query, doc, role="qd")  # query-doc
    sim, per_enc = pool.similarity(doc1, doc2, role="dd")  # doc-doc
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class EncoderSpec:
    name: str
    weight: float = 1.0
    query_prefix: str = ""
    passage_prefix: str = ""


class SingleEncoder:
    """Обёртка вокруг sentence-transformers модели."""

    def __init__(self, spec: EncoderSpec, device: str = "cuda",
                 max_length: int = 512):
        from sentence_transformers import SentenceTransformer
        self.spec = spec
        self.model = SentenceTransformer(spec.name, device=device)
        self.model.max_seq_length = max_length

    def encode(self, texts: list[str], role: str, batch_size: int = 32):
        """role: 'q' для запроса, 'd' для документа."""
        prefix = self.spec.query_prefix if role == "q" else self.spec.passage_prefix
        prefixed = [prefix + t for t in texts]
        embs = self.model.encode(
            prefixed,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embs

    def cos_sim(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Принимает уже нормированные эмбеддинги, возвращает построчно cos."""
        # a: (n, d), b: (n, d) -> (n,)
        if a.ndim == 1:
            a = a[None, :]
        if b.ndim == 1:
            b = b[None, :]
        return np.sum(a * b, axis=-1)


class EncoderPool:
    """Ансамбль энкодеров. Все методы возвращают (ансамблевая_оценка, per_encoder_dict)."""

    def __init__(self, encoders: list[SingleEncoder], batch_size: int = 32):
        if not encoders:
            raise ValueError("Нужен хотя бы один энкодер")
        self.encoders = encoders
        self.batch_size = batch_size
        total_w = sum(e.spec.weight for e in encoders)
        self.weights = [e.spec.weight / total_w for e in encoders]

    @classmethod
    def from_config(cls, cfg: dict) -> "EncoderPool":
        ensemble_cfg = cfg.get("ensemble", [])
        device = cfg.get("device", "cuda")
        max_length = cfg.get("max_length", 512)
        batch_size = cfg.get("batch_size", 32)
        encoders = [
            SingleEncoder(
                EncoderSpec(
                    name=e["name"],
                    weight=e.get("weight", 1.0),
                    query_prefix=e.get("query_prefix", ""),
                    passage_prefix=e.get("passage_prefix", ""),
                ),
                device=device,
                max_length=max_length,
            )
            for e in ensemble_cfg
        ]
        return cls(encoders, batch_size=batch_size)

    def similarity_pair(
        self, text_a: str, text_b: str, *, role: str = "qd"
    ) -> tuple[float, dict[str, float]]:
        """
        role:
            'qd' — text_a это query, text_b это document
            'dd' — оба документа (для GP, где сравниваем d⁺ vs dᵢ)
            'qq' — оба запроса
        """
        role_a, role_b = role[0], role[1]
        per_enc = {}
        agg = 0.0
        for enc, w in zip(self.encoders, self.weights):
            emb_a = enc.encode([text_a], role=role_a, batch_size=1)
            emb_b = enc.encode([text_b], role=role_b, batch_size=1)
            sim = float(enc.cos_sim(emb_a, emb_b)[0])
            per_enc[enc.spec.name] = sim
            agg += w * sim
        return agg, per_enc

    def similarity_batch(
        self,
        anchor: str,
        candidates: list[str],
        *,
        role: str = "qd",
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """
        Один anchor, много кандидатов. Эффективнее, чем similarity_pair в цикле.
        Возвращает (ансамблевые sim длины N, словарь per_encoder sim длины N).
        """
        role_a, role_b = role[0], role[1]
        per_enc = {}
        agg = np.zeros(len(candidates), dtype=np.float32)
        for enc, w in zip(self.encoders, self.weights):
            emb_a = enc.encode([anchor], role=role_a, batch_size=1)  # (1, d)
            emb_b = enc.encode(candidates, role=role_b,
                               batch_size=self.batch_size)             # (N, d)
            sim = (emb_b @ emb_a.T).squeeze(-1)  # (N,)
            per_enc[enc.spec.name] = sim
            agg += w * sim
        return agg, per_enc

    def delta(
        self, query: str, d_plus: str, d_minus: str
    ) -> tuple[float, dict[str, float]]:
        """
        δ = sim(q, d⁻) / sim(q, d⁺) 

        Считаем для каждого энкодера отдельно, потом усредняем δ (а НЕ
        усредняем числитель/знаменатель отдельно, чтобы избежать
        чувствительности к нормировке).
        """
        per_enc_delta = {}
        weighted_delta = 0.0
        for enc, w in zip(self.encoders, self.weights):
            emb_q = enc.encode([query], role="q", batch_size=1)
            emb_p = enc.encode([d_plus], role="d", batch_size=1)
            emb_n = enc.encode([d_minus], role="d", batch_size=1)
            sim_pos = float(enc.cos_sim(emb_q, emb_p)[0])
            sim_neg = float(enc.cos_sim(emb_q, emb_n)[0])
            # защита от деления на 0/нерелевантного d⁺
            if abs(sim_pos) < 1e-6:
                d = 0.0
            else:
                d = sim_neg / sim_pos
            per_enc_delta[enc.spec.name] = d
            weighted_delta += w * d
        return weighted_delta, per_enc_delta
