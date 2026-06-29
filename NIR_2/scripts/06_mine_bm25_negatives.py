#!/usr/bin/env python3
"""
Phase PoC / BM25: hard negatives mining через лексический ранкинг.

ЧТО ЭТО.
  Стандартный baseline для ablation grid (условие B): для каждой query из
  RuBQ найти top-K документов из корпуса с максимальным BM25 score,
  исключая собственный d⁺. Эти документы — "hard negatives" в лексическом
  смысле: они высоко-перекрываются с query по словам, но (предположительно)
  не являются правильным ответом.

ЗАЧЕМ.
  В ablation grid условие B = "+BM25 top-K hard" сравнивается с условием
  C = "+CN" (главная гипотеза работы). Если наш CN не побьёт это —
  гипотеза "контрастирование по факту лучше чем по топику" опровергнута.
  Поэтому baseline B обязан быть честной реализацией стандарта литературы.

ПРИНЦИПЫ.
  - Corpus = ВСЕ 2096 валидных d⁺ из RuBQ (не подмножество). Mining идёт
    во всём корпусе, независимо от того сколько queries обрабатываем.
  - НЕТ false-positive filter (CE/LLM). Стандарт литературы — top-K без
    фильтра. Если фильтровать, baseline становится "улучшенным" и теряет
    смысл как cleanly comparable условие.
  - Self-exclusion ПО qid, а не по тексту: если в corpus есть дубликаты
    d⁺ с другими qid (по факту они там есть после dedupe), они валидны
    как negatives — это feature BM25.
  - Dedupe corpus по нормализованному d_plus (защита от точных дублей).

ФОРМАТ ВЫХОДА.
  Flat-records (один на каждый retrieved negative), совместимый с CN/GP
  outputs, чтобы 07_build_train_data.py читал одинаково:
    {qid, query, d_plus, d_minus, bm25_score, rank, source_qid, method="bm25",
     passed_filters, tags}

ЗАПУСК.
  # Pilot:
  python scripts/06_mine_bm25_negatives.py --pilot

  # Production (500 queries, ~1-2 минуты — BM25 быстрый):
  python scripts/06_mine_bm25_negatives.py --n-examples 500

  # Шире (top-10 вместо top-5):
  python scripts/06_mine_bm25_negatives.py --n-examples 500 --k 10

ВЫХОД (в outputs/bm25_negatives/<mode>/):
  candidates_raw.jsonl        — все top-K результаты с диагностикой
  candidates_passed.jsonl     — прошедшие min-score фильтр
  summary.json
  report.md
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median, stdev

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


log = logging.getLogger("bm25_mining")


# ──────────────────────────────────────────────────────────────────────
# I/O
# ──────────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def save_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ──────────────────────────────────────────────────────────────────────
# Токенизация (тот же стек что в filters.py:BM25Filter)
# ──────────────────────────────────────────────────────────────────────

_FALLBACK_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _make_tokenizer():
    """razdel если доступен, иначе regex fallback. Возвращает callable str → list[str]."""
    try:
        from razdel import tokenize as razdel_tok
        log.info("Tokenizer: razdel")
        return lambda t: [x.text.lower() for x in razdel_tok(t) if x.text.isalnum()]
    except ImportError:
        log.warning("razdel недоступен — fallback на regex \\w+")
        return lambda t: _FALLBACK_TOKEN_RE.findall(t.lower())


# ──────────────────────────────────────────────────────────────────────
# Corpus загрузка + dedupe
# ──────────────────────────────────────────────────────────────────────

def _normalize_for_dedupe(text: str) -> str:
    """Грубая нормализация для dedupe: lowercase + сжатие whitespace."""
    return re.sub(r"\s+", " ", text.strip().lower())


def load_corpus(path: Path) -> tuple[list[dict], int]:
    """
    Загружает rubq_full, выполняет dedupe по нормализованному d_plus.
    Возвращает (corpus_docs, n_duplicates_removed).

    Каждый corpus_doc: {qid, d_plus, query (исходный), tags}.
    """
    raw = load_jsonl(path)
    seen: dict[str, dict] = {}
    n_dup = 0
    for ex in raw:
        key = _normalize_for_dedupe(ex["d_plus"])
        if key in seen:
            n_dup += 1
            continue
        seen[key] = {
            "qid": ex["qid"],
            "d_plus": ex["d_plus"],
            "query": ex.get("query", ""),
            "tags": ex.get("meta", {}).get("tags", []),
        }
    return list(seen.values()), n_dup


# ──────────────────────────────────────────────────────────────────────
# Sampling (тот же стиль что в 04 и 05)
# ──────────────────────────────────────────────────────────────────────

def sample_queries(all_examples: list[dict], n: int, seed: int = 42) -> list[dict]:
    if n >= len(all_examples):
        return all_examples
    rng = random.Random(seed)
    by_tag = defaultdict(list)
    for ex in all_examples:
        tags = ex.get("meta", {}).get("tags") or []
        primary = tags[0] if tags else "untagged"
        by_tag[primary].append(ex)

    total = len(all_examples)
    out = []
    for tag, group in by_tag.items():
        rng.shuffle(group)
        take = max(1, round(n * len(group) / total))
        out.extend(group[:take])
    rng.shuffle(out)
    return out[:n]


# ──────────────────────────────────────────────────────────────────────
# BM25 wrapper (rank_bm25)
# ──────────────────────────────────────────────────────────────────────

class BM25Index:
    """
    Тонкая обёртка над rank_bm25.BM25Okapi.

    Хранит токенизированный corpus и предоставляет top-K query с
    исключением заданных qid (self-exclusion).
    """

    def __init__(self, corpus_docs: list[dict], tokenizer):
        try:
            from rank_bm25 import BM25Okapi  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "rank_bm25 не установлен. Установите: pip install rank_bm25"
            ) from e
        from rank_bm25 import BM25Okapi

        self.docs = corpus_docs
        self.tokenizer = tokenizer
        log.info("Токенизирую corpus (%d документов)...", len(corpus_docs))
        self.tokenized = [tokenizer(d["d_plus"]) for d in corpus_docs]
        log.info("Строю BM25 индекс...")
        self.bm25 = BM25Okapi(self.tokenized)
        # qid → corpus_index для быстрого self-exclusion
        self.qid_to_idx = {d["qid"]: i for i, d in enumerate(corpus_docs)}

    def top_k(
        self,
        query: str,
        k: int,
        exclude_qid: str | None = None,
        min_score: float = 0.0,
    ) -> list[dict]:
        """
        Возвращает список dict с полями {corpus_idx, qid, d_plus, bm25_score, rank, tags}.
        rank — 1-indexed позиция в исходном top без exclude.
        """
        q_tokens = self.tokenizer(query)
        if not q_tokens:
            return []
        scores = self.bm25.get_scores(q_tokens)
        # argsort desc
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        results = []
        rank = 0
        for idx in order:
            rank += 1
            score = float(scores[idx])
            doc = self.docs[idx]
            if exclude_qid is not None and doc["qid"] == exclude_qid:
                continue
            if score < min_score:
                break  # дальше будет только меньше
            results.append({
                "corpus_idx": idx,
                "qid": doc["qid"],
                "d_plus": doc["d_plus"],
                "bm25_score": score,
                "rank": rank,
                "tags": doc.get("tags", []),
            })
            if len(results) >= k:
                break
        return results


# ──────────────────────────────────────────────────────────────────────
# Mining loop
# ──────────────────────────────────────────────────────────────────────

def mine_query(
    bm25: BM25Index,
    ex: dict,
    *,
    k: int,
    min_score: float,
) -> tuple[list[dict], dict]:
    """
    Возвращает (records, diag).
      records: top-K negatives как flat-records, готовые к записи.
      diag: diagnostic info по этой query.
    """
    hits = bm25.top_k(ex["query"], k=k, exclude_qid=ex["qid"], min_score=min_score)

    # Self-match диагностика: top-1 ИСХОДНЫЙ rank до exclude. Если qid_to_idx
    # совпадает с первой позицией без exclude — это значит BM25 распознал d⁺
    # как top-1 (хорошо: signal sanity).
    self_idx = bm25.qid_to_idx.get(ex["qid"])
    self_was_top1 = False
    self_in_top10 = False
    if self_idx is not None:
        scores = bm25.bm25.get_scores(bm25.tokenizer(ex["query"]))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        if order and order[0] == self_idx:
            self_was_top1 = True
        if self_idx in order[:10]:
            self_in_top10 = True

    records = []
    for h in hits:
        records.append({
            "qid": ex["qid"],                    # source query
            "query": ex["query"],
            "d_plus": ex["d_plus"],
            "d_minus": h["d_plus"],              # mined negative
            "source_qid": h["qid"],              # откуда негатив в corpus
            "bm25_score": h["bm25_score"],
            "rank": h["rank"],
            "method": "bm25",
            "passed_filters": True,              # для consistency с CN/GP
            "rejection_reason": None,
            "tags": ex.get("meta", {}).get("tags", []),
            "neg_tags": h.get("tags", []),
        })

    diag = {
        "n_hits": len(hits),
        "self_was_top1": self_was_top1,
        "self_in_top10": self_in_top10,
        "max_score": hits[0]["bm25_score"] if hits else None,
        "min_score_in_topk": hits[-1]["bm25_score"] if hits else None,
    }
    return records, diag


# ──────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────

def _percentiles(xs: list[float], qs=(10, 25, 50, 75, 90)) -> dict:
    if not xs:
        return {f"p{q}": None for q in qs}
    s = sorted(xs)
    n = len(s)
    out = {}
    for q in qs:
        idx = max(0, min(n - 1, int(round(q / 100 * (n - 1)))))
        out[f"p{q}"] = round(s[idx], 4)
    return out


def compute_summary(
    records: list[dict],
    diags_by_qid: dict[str, dict],
    examples: list[dict],
    corpus_size: int,
    n_duplicates: int,
    *,
    k: int,
    min_score: float,
) -> dict:
    n_input = len(examples)
    qids_with_any = {r["qid"] for r in records}

    # Покрытие: сколько queries получили хотя бы 1 negative
    n_with_neg = len(qids_with_any)
    # И сколько получили ровно k (полный комплект)
    counts_per_qid = Counter(r["qid"] for r in records)
    n_with_full_k = sum(1 for c in counts_per_qid.values() if c >= k)

    # BM25 score stats
    all_scores = [r["bm25_score"] for r in records]
    top1_scores = [r["bm25_score"] for r in records if r["rank"] == 1
                   or r["rank"] == _first_rank_per_qid(records, r["qid"])]
    # Более явный способ: top1 = первая запись на каждый qid (по позиции в списке)
    seen_qid = set()
    top1_scores = []
    for r in records:
        if r["qid"] not in seen_qid:
            top1_scores.append(r["bm25_score"])
            seen_qid.add(r["qid"])

    # Self-match diagnostic
    n_self_top1 = sum(1 for d in diags_by_qid.values() if d.get("self_was_top1"))
    n_self_in_top10 = sum(1 for d in diags_by_qid.values() if d.get("self_in_top10"))

    summary = {
        "config": {
            "k": k,
            "min_score": min_score,
        },
        "corpus": {
            "n_docs": corpus_size,
            "n_duplicates_removed": n_duplicates,
        },
        "n_input_queries": n_input,
        "coverage": {
            "n_query_with_any_negative": n_with_neg,
            "n_query_with_full_k": n_with_full_k,
            "pct_with_full_k": round(100 * n_with_full_k / max(1, n_input), 1),
        },
        "negatives": {
            "n_total": len(records),
            "expected_total": n_input * k,
            "fill_rate": round(len(records) / max(1, n_input * k), 4),
        },
        "bm25_scores_top1": {
            "n": len(top1_scores),
            "mean":   round(mean(top1_scores), 4) if top1_scores else None,
            "median": round(median(top1_scores), 4) if top1_scores else None,
            "stdev":  round(stdev(top1_scores), 4) if len(top1_scores) > 1 else None,
            "min":    round(min(top1_scores), 4) if top1_scores else None,
            "max":    round(max(top1_scores), 4) if top1_scores else None,
            **_percentiles(top1_scores),
        },
        "bm25_scores_all": {
            "n": len(all_scores),
            "mean":   round(mean(all_scores), 4) if all_scores else None,
            "median": round(median(all_scores), 4) if all_scores else None,
            **_percentiles(all_scores),
        },
        "self_match_diagnostic": {
            "n_self_was_top1": n_self_top1,
            "pct_self_was_top1": round(100 * n_self_top1 / max(1, n_input), 1),
            "n_self_in_top10": n_self_in_top10,
            "pct_self_in_top10": round(100 * n_self_in_top10 / max(1, n_input), 1),
            "note": "self_was_top1 high (>50%) → BM25 хорошо работает как retriever "
                    "(надёжный сигнал что d⁺ лексически релевантен q). "
                    "self_was_top1 low → query слабо пересекается с d⁺ по словам "
                    "(множество multi-hop / переформулированных queries в RuBQ).",
        },
    }
    return summary


def _first_rank_per_qid(records, qid):
    """Helper: returns rank of first record matching qid in the list order."""
    for r in records:
        if r["qid"] == qid:
            return r["rank"]
    return None


# ──────────────────────────────────────────────────────────────────────
# Report.md (КОНТРАКТ)
# ──────────────────────────────────────────────────────────────────────

def write_report(
    summary: dict,
    records: list[dict],
    examples: list[dict],
    output_dir: Path,
    mode_name: str,
) -> None:
    cfg = summary["config"]
    corp = summary["corpus"]
    cov = summary["coverage"]
    neg = summary["negatives"]
    s_top1 = summary["bm25_scores_top1"]
    s_all = summary["bm25_scores_all"]
    sm = summary["self_match_diagnostic"]

    lines = []
    lines.append(f"# BM25 Hard Negatives Mining — {mode_name}\n")
    lines.append(f"- Corpus size: **{corp['n_docs']}** "
                 f"(removed {corp['n_duplicates_removed']} duplicates)")
    lines.append(f"- Input queries: **{summary['n_input_queries']}**")
    lines.append(f"- k (negatives per query): **{cfg['k']}**, "
                 f"min_score = {cfg['min_score']}")
    lines.append(f"- Coverage: **{cov['n_query_with_full_k']}/"
                 f"{summary['n_input_queries']}** "
                 f"({cov['pct_with_full_k']}%) с полным top-{cfg['k']}")
    lines.append(f"- Negatives total: **{neg['n_total']}** "
                 f"(expected {neg['expected_total']}, fill rate {neg['fill_rate']:.1%})")
    lines.append("")

    # BM25 score distribution
    lines.append("## BM25 score distribution\n")
    if s_top1["n"]:
        stdev_str = f"{s_top1['stdev']:.4f}" if s_top1["stdev"] is not None else "n/a"
        lines.append(f"**Top-1 scores per query (n={s_top1['n']}):**")
        lines.append(f"- mean={s_top1['mean']:.4f}  median={s_top1['median']:.4f}  "
                     f"stdev={stdev_str}")
        lines.append(f"- range=[{s_top1['min']:.4f}, {s_top1['max']:.4f}]")
        lines.append(f"- p10={s_top1['p10']}  p25={s_top1['p25']}  "
                     f"p50={s_top1['p50']}  p75={s_top1['p75']}  p90={s_top1['p90']}")
        lines.append("")
    if s_all["n"]:
        lines.append(f"**All scores (n={s_all['n']}):**")
        lines.append(f"- mean={s_all['mean']:.4f}  median={s_all['median']:.4f}")
        lines.append(f"- p10={s_all['p10']}  p50={s_all['p50']}  p90={s_all['p90']}")
    lines.append("")

    # Self-match diagnostic
    lines.append("## Self-match diagnostic (BM25 как retriever sanity check)\n")
    lines.append(f"- d⁺ был top-1 в BM25 ранкинге: **{sm['n_self_was_top1']}/"
                 f"{summary['n_input_queries']}** ({sm['pct_self_was_top1']}%)")
    lines.append(f"- d⁺ был в top-10: **{sm['n_self_in_top10']}/"
                 f"{summary['n_input_queries']}** ({sm['pct_self_in_top10']}%)")
    lines.append(f"\n> {sm['note']}")
    lines.append("")

    # Примеры
    lines.append("---\n## Примеры извлечённых negatives\n")
    qd_map = {ex["qid"]: ex for ex in examples}
    by_qid = defaultdict(list)
    for r in records:
        by_qid[r["qid"]].append(r)

    for qid in list(by_qid.keys())[:3]:
        ex = qd_map.get(qid)
        if not ex:
            continue
        lines.append(f"### {qid}\n")
        lines.append(f"**Query:** {ex['query']}\n")
        dp = ex["d_plus"][:220].replace("\n", " ")
        lines.append(f"**d⁺:** {dp}{'…' if len(ex['d_plus']) > 220 else ''}\n")
        lines.append(f"**Top-{cfg['k']} BM25 negatives:**\n")
        for r in sorted(by_qid[qid], key=lambda x: x["rank"]):
            dm = r["d_minus"][:200].replace("\n", " ")
            lines.append(f"- **rank {r['rank']}** "
                         f"(score={r['bm25_score']:.3f}, "
                         f"source={r['source_qid']}):")
            lines.append(f"  > {dm}{'…' if len(r['d_minus']) > 200 else ''}")
        lines.append("")

    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(PROJECT_ROOT / "data/rubq/rubq_full.jsonl"),
                    help="JSONL с q/d⁺ парами; используется и как corpus, и как пул query")
    ap.add_argument("--corpus", default=None,
                    help="отдельный JSONL для corpus (если не указан = --input)")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--n-examples", type=int, default=500)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--k", type=int, default=5,
                    help="negatives на query (default: 5)")
    ap.add_argument("--min-score", type=float, default=0.0,
                    help="порог BM25 score; ниже отбрасываем (default: 0.0 — берём всё)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )

    mode_name = "pilot" if args.pilot else "prod"
    n = 10 if args.pilot else args.n_examples
    out_dir = Path(args.output_dir) if args.output_dir else \
              (PROJECT_ROOT / "outputs" / "bm25_negatives" / mode_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Режим: %s, output: %s", mode_name, out_dir)
    log.info("k=%d  min_score=%.2f", args.k, args.min_score)

    # 1. Загрузка corpus (полный) + dedupe
    corpus_path = Path(args.corpus) if args.corpus else Path(args.input)
    log.info("Загружаю corpus: %s", corpus_path)
    corpus_docs, n_dup = load_corpus(corpus_path)
    log.info("Corpus: %d уникальных документов (removed %d дублей)",
             len(corpus_docs), n_dup)

    # 2. Загрузка query пула
    all_examples = load_jsonl(Path(args.input))
    log.info("Query pool: %d пар", len(all_examples))
    queries = sample_queries(all_examples, n, seed=args.seed)
    log.info("К обработке: %d query", len(queries))

    # 3. Индекс
    tokenizer = _make_tokenizer()
    index = BM25Index(corpus_docs, tokenizer)
    log.info("BM25 индекс готов")

    # 4. Mining loop
    try:
        from tqdm import tqdm
        pbar = tqdm(queries, desc="BM25 mining", unit="q")
    except ImportError:
        pbar = queries

    all_records: list[dict] = []
    diags_by_qid: dict[str, dict] = {}
    for ex in pbar:
        recs, diag = mine_query(
            index, ex,
            k=args.k, min_score=args.min_score,
        )
        all_records.extend(recs)
        diags_by_qid[ex["qid"]] = diag
        if args.pilot:
            top1 = diag.get("max_score")
            top1_str = f"{top1:.3f}" if top1 is not None else "—"
            log.info("[%s] hits=%d  top1_score=%s  self_top1=%s",
                     ex["qid"], diag["n_hits"], top1_str,
                     "Y" if diag["self_was_top1"] else "N")

    # 5. Запись
    save_jsonl(out_dir / "candidates_raw.jsonl", all_records)
    save_jsonl(out_dir / "candidates_passed.jsonl",
               [r for r in all_records if r["passed_filters"]])

    summary = compute_summary(
        all_records, diags_by_qid, queries,
        corpus_size=len(corpus_docs), n_duplicates=n_dup,
        k=args.k, min_score=args.min_score,
    )
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(summary, all_records, queries, out_dir, mode_name)

    # ──────────────────────────────────────────────
    # Печать сводки
    # ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"BM25 NEGATIVES — {mode_name.upper()}")
    print("=" * 70)
    cov = summary["coverage"]
    neg = summary["negatives"]
    s1 = summary["bm25_scores_top1"]
    sm = summary["self_match_diagnostic"]

    print(f"Corpus:                {summary['corpus']['n_docs']} docs "
          f"(removed {summary['corpus']['n_duplicates_removed']} dups)")
    print(f"Input queries:         {summary['n_input_queries']}")
    print(f"Full top-{args.k} coverage:  "
          f"{cov['n_query_with_full_k']}/{summary['n_input_queries']} "
          f"({cov['pct_with_full_k']}%)")
    print(f"Negatives total:       {neg['n_total']} / {neg['expected_total']} "
          f"({neg['fill_rate']:.1%} fill rate)")
    print()
    if s1["n"]:
        stdev_str = f"{s1['stdev']:.4f}" if s1["stdev"] is not None else "n/a"
        print(f"BM25 top-1 scores (n={s1['n']}):")
        print(f"  mean={s1['mean']:.4f}  median={s1['median']:.4f}  stdev={stdev_str}")
        print(f"  range=[{s1['min']:.4f}, {s1['max']:.4f}]  "
              f"p10={s1['p10']}  p90={s1['p90']}")
    print()
    print(f"d⁺ как top-1 в BM25:    {sm['n_self_was_top1']}/{summary['n_input_queries']} "
          f"({sm['pct_self_was_top1']}%)")
    print(f"d⁺ в top-10 в BM25:     {sm['n_self_in_top10']}/{summary['n_input_queries']} "
          f"({sm['pct_self_in_top10']}%)")
    print()
    print("Files:")
    print(f"  {out_dir / 'candidates_raw.jsonl'}")
    print(f"  {out_dir / 'candidates_passed.jsonl'}")
    print(f"  {out_dir / 'summary.json'}")
    print(f"  {out_dir / 'report.md'}")

    if args.pilot:
        print("\n" + "─" * 70)
        print("PILOT CHECKLIST (BM25)")
        print("─" * 70)
        # Ожидания:
        #  - Full coverage 100% (BM25 всегда найдёт top-k если corpus достаточный)
        #  - Top-1 mean > 5.0 (типичный BM25 score, зависит от длин)
        #  - d⁺ как top-1 ≥ 30% (sanity: BM25 находит d⁺ для нормальных query.
        #    Если меньше — query очень короткие или сильно переформулированные)
        checks = [
            ("Full top-k coverage = 100%",
             cov["pct_with_full_k"] >= 100.0,
             f"{cov['pct_with_full_k']}%"),
            ("Fill rate = 100%",
             neg["fill_rate"] >= 1.0,
             f"{neg['fill_rate']:.1%}"),
            ("Top-1 BM25 mean > 1.0 (есть лексический сигнал)",
             s1["mean"] is not None and s1["mean"] > 1.0,
             f"{s1['mean']}"),
            ("d⁺ как BM25 top-1 ≥ 30% (BM25 sanity)",
             sm["pct_self_was_top1"] >= 30.0,
             f"{sm['pct_self_was_top1']}%"),
        ]
        for name, ok, val in checks:
            mark = "✓" if ok else "✗"
            print(f"  {mark}  {name}  →  {val}")
        if all(c[1] for c in checks):
            print("\n✓ Pilot OK — запускай production: --n-examples 500 (без --pilot)")
        else:
            print("\n⚠ Есть провалы.")
            print("  - Full coverage < 100% → corpus слишком маленький или min-score слишком высокий")
            print("  - d⁺ как top-1 низкий → multi-hop queries не пересекаются с d⁺ по словам;")
            print("    это feature RuBQ (не bug), но baseline BM25 будет слабее.")


if __name__ == "__main__":
    main()
