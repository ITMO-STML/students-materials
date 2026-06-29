#!/usr/bin/env python3
"""
Phase PoC: сборка train data для 5 условий ablation grid.

ВХОД.
  - data/rubq/rubq_full.jsonl                                — все 2096 пар.
  - outputs/cn_pipeline/prod/candidates_passed.jsonl         — CN negatives.
  - outputs/gp_pipeline_iter/prod/candidates_passed.jsonl    — GP positives.
  - outputs/bm25_negatives/prod/candidates_passed.jsonl      — BM25 negatives.

ВЫХОД (data/train/).
  split.json                — детерминированный train/test split (seed=42).
  A_baseline.jsonl          — train для условия A (in-batch only).
  B_bm25_hn.jsonl           — train для B (+ BM25 hard negatives).
  C_cn.jsonl                — train для C (+ CN hard negatives).
  D_gp.jsonl                — train для D (d⁺ + GP перефразы).
  E_cn_gp.jsonl             — train для E (CN HN + GP перефразы).
  test.jsonl                — общий test (200 queries) с gold_qid.
  corpus.jsonl              — все 2096 d⁺ для retrieval eval.
  summary.json
  report.md

ABLATION GRID.

  | # | Positives             | Negatives        |
  |---|-----------------------|------------------|
  | A | d⁺                    | in-batch only    |
  | B | d⁺                    | BM25 top-K       |
  | C | d⁺                    | CN top-K         |
  | D | d⁺ + GP перефразы     | in-batch only    |
  | E | d⁺ + GP перефразы     | CN top-K         |

  ВО ВСЕХ УСЛОВИЯХ: tokenizer, optimizer, lr, batch size, epochs, seed — ОДИНАКОВЫЕ.
  Меняется ТОЛЬКО train data. Это и есть чистый ablation.

DESIGN DECISIONS.
  1. Если для query нет CN/BM25 negatives — пара (q, d⁺) идёт в train БЕЗ
     hard negatives. Модель учится только на in-batch. НЕ заменяем
     BM25 fallback'ом в условии C — иначе смазали бы эффект.
  2. Если для query нет GP variants — только оригинальный d⁺ (degraded augm).
  3. Cap на число negatives (default 3) — для cleaner comparison между B/C/E.
  4. Train/test split стратифицирован по primary tag.

ФОРМАТ ЗАПИСЕЙ.
  Стандарт sentence-transformers (MultipleNegativesRankingLoss):
    {qid, query, positive, negatives: [str, ...]}

  Для D/E с augmentation — дублируем записи (одна запись = один positive).
  Поле "source_variant" различает: original | gp_iter_k_chain_c.

ЗАПУСК.
  python scripts/07_build_train_data.py
  python scripts/07_build_train_data.py --n-train 500 --n-test 200 --cap-negatives 3
  python scripts/07_build_train_data.py --conditions A B C   # только подмножество
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

log = logging.getLogger("build_train")


# ──────────────────────────────────────────────────────────────────────
# I/O
# ──────────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
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
# Train/test split (стратифицированный по primary tag, детерминированный)
# ──────────────────────────────────────────────────────────────────────

def _primary_tag(ex: dict) -> str:
    tags = ex.get("meta", {}).get("tags") or []
    return tags[0] if tags else "untagged"


def make_train_test_split(
    all_examples: list[dict],
    n_train: int,
    n_test: int,
    seed: int = 42,
) -> tuple[list[dict], list[dict], dict]:
    """
    Детерминированный стратифицированный split.

    Возвращает (train, test, diag).
      train, test: списки QueryDoc-словарей.
      diag: распределение primary_tag в train/test, edge cases.

    Стратегия. Группируем по primary tag (1-hop / multi-constraint / multi-hop /
    other), внутри тега shuffle с фиксированным seed, берём долю
    пропорционально размеру тега в исходном пуле. Гарантирует что и в train,
    и в test есть представители всех тегов.
    """
    if n_train + n_test > len(all_examples):
        raise ValueError(
            f"n_train ({n_train}) + n_test ({n_test}) > "
            f"corpus ({len(all_examples)})"
        )

    rng = random.Random(seed)
    by_tag: dict[str, list[dict]] = defaultdict(list)
    for ex in all_examples:
        by_tag[_primary_tag(ex)].append(ex)

    total = len(all_examples)
    train, test = [], []
    train_share, test_share = n_train / total, n_test / total

    for tag, group in by_tag.items():
        rng.shuffle(group)
        n_tr = round(len(group) * train_share)
        n_te = round(len(group) * test_share)
        # Защита от переполнения внутри тега
        n_tr = min(n_tr, len(group))
        n_te = min(n_te, len(group) - n_tr)
        train.extend(group[:n_tr])
        test.extend(group[n_tr:n_tr + n_te])

    # Финальный shuffle: чтобы порядок не был "все 1-hop потом всё остальное"
    rng.shuffle(train)
    rng.shuffle(test)

    # Если из-за округления получили больше/меньше — корректируем
    train = train[:n_train]
    test = test[:n_test]

    diag = {
        "n_train": len(train),
        "n_test": len(test),
        "n_unused": total - len(train) - len(test),
        "train_tags": dict(Counter(_primary_tag(e) for e in train)),
        "test_tags":  dict(Counter(_primary_tag(e) for e in test)),
    }

    # Sanity: disjoint qids
    qids_train = {e["qid"] for e in train}
    qids_test = {e["qid"] for e in test}
    overlap = qids_train & qids_test
    if overlap:
        raise RuntimeError(f"train/test overlap: {len(overlap)} qids")

    return train, test, diag


# ──────────────────────────────────────────────────────────────────────
# Адаптеры: извлекаем (qid, doc_text) из записей разного формата
# ──────────────────────────────────────────────────────────────────────

def _cn_text(rec: dict) -> str | None:
    """CN запись (формат 05): поле d_minus."""
    t = rec.get("d_minus")
    return t.strip() if isinstance(t, str) and t.strip() else None


def _bm25_text(rec: dict) -> str | None:
    """BM25 запись (формат 06): поле d_minus."""
    t = rec.get("d_minus")
    return t.strip() if isinstance(t, str) and t.strip() else None


def _gp_text(rec: dict) -> str | None:
    """GP запись (формат 04 итеративный): поле text."""
    t = rec.get("text") or rec.get("d_plus_i")  # на случай старого формата
    return t.strip() if isinstance(t, str) and t.strip() else None


def _gp_passed(rec: dict) -> bool:
    """
    Прошла ли GP запись фильтры. В 04_run_gp_pipeline нет явного
    passed_filters — есть llm_verdict ('yes'/'no'/None) и stop_reason.
    Считаем passed если verdict=='yes' и stop_reason in (None, '').
    """
    if "passed_filters" in rec:  # на будущее
        return bool(rec["passed_filters"])
    v = rec.get("llm_verdict")
    stop = rec.get("stop_reason")
    return v == "yes" and not stop


# ──────────────────────────────────────────────────────────────────────
# Группировка negatives/positives по qid
# ──────────────────────────────────────────────────────────────────────

def group_negatives(
    records: list[dict],
    text_fn,
    *,
    label: str,
) -> dict[str, list[str]]:
    """Группирует d⁻ по source qid. Дедуп внутри одной query."""
    out: dict[str, list[str]] = defaultdict(list)
    n_skipped = 0
    for r in records:
        # Опциональный фильтр (CN/BM25 уже передают только passed)
        if "passed_filters" in r and not r["passed_filters"]:
            n_skipped += 1
            continue
        qid = r.get("qid")
        text = text_fn(r)
        if not qid or not text:
            n_skipped += 1
            continue
        if text not in out[qid]:  # dedup в пределах одного qid
            out[qid].append(text)
    log.info("[%s] grouped: %d qids with negatives, %d records skipped",
             label, len(out), n_skipped)
    return dict(out)


def group_gp_positives(
    records: list[dict],
) -> dict[str, list[dict]]:
    """
    Группирует GP positives по qid. Сохраняем доп. поля
    (iteration, chain_idx, sim_to_d_plus) для source_variant и диагностики.
    Только passed (verdict=yes, без stop_reason).
    """
    out: dict[str, list[dict]] = defaultdict(list)
    n_skipped = 0
    for r in records:
        if not _gp_passed(r):
            n_skipped += 1
            continue
        qid = r.get("qid")
        text = _gp_text(r)
        if not qid or not text:
            n_skipped += 1
            continue
        # Дедуп по тексту
        if any(p["text"] == text for p in out[qid]):
            continue
        out[qid].append({
            "text": text,
            "iter": r.get("iteration"),
            "chain": r.get("chain_idx"),
            "sim_to_d_plus": r.get("sim_to_d_plus"),
        })
    log.info("[GP] grouped: %d qids with positives, %d records skipped",
             len(out), n_skipped)
    return dict(out)


# ──────────────────────────────────────────────────────────────────────
# Builders: каждое условие — отдельная функция
# ──────────────────────────────────────────────────────────────────────

def build_A(train_exs: list[dict]) -> list[dict]:
    """A: in-batch only. (q, d⁺) пары, negatives=[]."""
    return [
        {
            "qid": ex["qid"],
            "query": ex["query"],
            "positive": ex["d_plus"],
            "negatives": [],
            "source_variant": "original",
        }
        for ex in train_exs
    ]


def build_B(
    train_exs: list[dict],
    bm25_neg: dict[str, list[str]],
    *,
    cap_negatives: int,
) -> list[dict]:
    """B: d⁺ + BM25 top-K hard negatives."""
    out = []
    for ex in train_exs:
        negs = bm25_neg.get(ex["qid"], [])[:cap_negatives]
        out.append({
            "qid": ex["qid"],
            "query": ex["query"],
            "positive": ex["d_plus"],
            "negatives": negs,
            "source_variant": "original",
        })
    return out


def build_C(
    train_exs: list[dict],
    cn_neg: dict[str, list[str]],
    *,
    cap_negatives: int,
) -> list[dict]:
    """C: d⁺ + CN hard negatives. Если CN не нашёл — negatives=[] (НЕ fallback на BM25)."""
    out = []
    for ex in train_exs:
        negs = cn_neg.get(ex["qid"], [])[:cap_negatives]
        out.append({
            "qid": ex["qid"],
            "query": ex["query"],
            "positive": ex["d_plus"],
            "negatives": negs,
            "source_variant": "original",
        })
    return out


def build_D(
    train_exs: list[dict],
    gp_pos: dict[str, list[dict]],
) -> list[dict]:
    """
    D: in-batch only + augmentation. Для каждой query:
      - одна запись с оригинальным d⁺
      - дополнительные записи с GP-вариантами
    Записи помечаются source_variant.
    """
    out = []
    for ex in train_exs:
        # Оригинал
        out.append({
            "qid": ex["qid"],
            "query": ex["query"],
            "positive": ex["d_plus"],
            "negatives": [],
            "source_variant": "original",
        })
        # GP варианты
        for p in gp_pos.get(ex["qid"], []):
            out.append({
                "qid": ex["qid"],
                "query": ex["query"],
                "positive": p["text"],
                "negatives": [],
                "source_variant": f"gp_iter{p.get('iter')}_chain{p.get('chain')}",
            })
    return out


def build_E(
    train_exs: list[dict],
    cn_neg: dict[str, list[str]],
    gp_pos: dict[str, list[dict]],
    *,
    cap_negatives: int,
) -> list[dict]:
    """E: CN HN + GP positives. Композиция C+D."""
    out = []
    for ex in train_exs:
        negs = cn_neg.get(ex["qid"], [])[:cap_negatives]
        # Оригинал с CN
        out.append({
            "qid": ex["qid"],
            "query": ex["query"],
            "positive": ex["d_plus"],
            "negatives": negs,
            "source_variant": "original",
        })
        # GP-варианты тоже с CN negatives (тот же набор)
        for p in gp_pos.get(ex["qid"], []):
            out.append({
                "qid": ex["qid"],
                "query": ex["query"],
                "positive": p["text"],
                "negatives": negs,
                "source_variant": f"gp_iter{p.get('iter')}_chain{p.get('chain')}",
            })
    return out


# ──────────────────────────────────────────────────────────────────────
# Corpus + test
# ──────────────────────────────────────────────────────────────────────

def build_corpus(all_examples: list[dict]) -> list[dict]:
    """
    Corpus для retrieval eval: каждый d⁺ как документ с уникальным qid.
    Dedupe по нормализованному тексту (как в 06).
    """
    import re
    seen = {}
    n_dup = 0
    for ex in all_examples:
        key = re.sub(r"\s+", " ", ex["d_plus"].strip().lower())
        if key in seen:
            n_dup += 1
            continue
        seen[key] = {"qid": ex["qid"], "text": ex["d_plus"]}
    log.info("Corpus: %d уникальных docs (removed %d duplicates)",
             len(seen), n_dup)
    return list(seen.values())


def build_test(test_exs: list[dict], corpus: list[dict]) -> list[dict]:
    """
    Test set: {qid, query, gold_qid}. gold_qid — qid в corpus у правильного d⁺.

    Для каждой test query ищем её d⁺ в corpus по тексту (а не по qid):
    после dedupe в corpus один документ может иметь чужой qid (если был
    выбран как первый при dedupe). Главное чтобы тот же текст был в corpus.
    """
    import re
    # Индекс corpus по нормализованному тексту
    norm = {re.sub(r"\s+", " ", c["text"].strip().lower()): c["qid"] for c in corpus}
    out = []
    n_missing = 0
    for ex in test_exs:
        key = re.sub(r"\s+", " ", ex["d_plus"].strip().lower())
        gold = norm.get(key)
        if not gold:
            log.warning("test qid=%s: d⁺ не найден в corpus", ex["qid"])
            n_missing += 1
            continue
        out.append({
            "qid": ex["qid"],
            "query": ex["query"],
            "gold_qid": gold,
            "d_plus": ex["d_plus"],  # для удобства eval
        })
    if n_missing:
        log.warning("test: %d записей без gold (skipped)", n_missing)
    return out


# ──────────────────────────────────────────────────────────────────────
# Stats
# ──────────────────────────────────────────────────────────────────────

def stats_for_condition(records: list[dict], train_qids: set) -> dict:
    """Метрики для одного condition: n_records, neg coverage, augm rate."""
    n = len(records)
    by_qid_neg = defaultdict(list)
    by_qid_var = defaultdict(list)
    for r in records:
        by_qid_neg[r["qid"]].append(len(r["negatives"]))
        by_qid_var[r["qid"]].append(r["source_variant"])

    qids = set(by_qid_neg.keys())
    n_with_any_neg = sum(1 for qid in qids if max(by_qid_neg[qid], default=0) > 0)
    n_no_neg = len(qids) - n_with_any_neg

    all_neg_counts = [c for counts in by_qid_neg.values() for c in counts]
    all_var_counts = [len(set(v)) for v in by_qid_var.values()]

    return {
        "n_records": n,
        "n_unique_qids": len(qids),
        "n_qids_with_any_negative": n_with_any_neg,
        "n_qids_without_negatives": n_no_neg,
        "pct_qids_with_negatives": round(
            100 * n_with_any_neg / max(1, len(qids)), 1
        ),
        "mean_negatives_per_record": round(
            mean(all_neg_counts) if all_neg_counts else 0, 2
        ),
        "median_negatives_per_record": median(all_neg_counts) if all_neg_counts else 0,
        "max_negatives_per_record": max(all_neg_counts, default=0),
        "mean_variants_per_qid": round(
            mean(all_var_counts) if all_var_counts else 0, 2
        ),
        "max_variants_per_qid": max(all_var_counts, default=0),
        "missing_qids": sorted(train_qids - qids),  # должно быть пусто
    }


def compute_summary(
    split_diag: dict,
    corpus: list[dict],
    test_set: list[dict],
    condition_stats: dict[str, dict],
    raw_counts: dict[str, int],
    intersections: dict[str, int],
    *,
    cap_negatives: int,
) -> dict:
    return {
        "config": {
            "cap_negatives": cap_negatives,
        },
        "split": split_diag,
        "corpus": {
            "n_docs": len(corpus),
        },
        "test": {
            "n_queries": len(test_set),
        },
        "raw_inputs": raw_counts,
        "intersections": intersections,
        "conditions": condition_stats,
    }


# ──────────────────────────────────────────────────────────────────────
# Report.md (КОНТРАКТ)
# ──────────────────────────────────────────────────────────────────────

def write_report(
    summary: dict,
    output_dir: Path,
    train_exs: list[dict],
    test_set: list[dict],
) -> None:
    s = summary
    lines = []
    lines.append(f"# Train Data Build — Ablation Grid\n")
    lines.append(f"- Cap negatives per record: **{s['config']['cap_negatives']}**")
    lines.append(f"- Train queries: **{s['split']['n_train']}**, "
                 f"test queries: **{s['split']['n_test']}**, "
                 f"unused: {s['split']['n_unused']}")
    lines.append(f"- Corpus size: **{s['corpus']['n_docs']}** docs")
    lines.append(f"- Test set written: **{s['test']['n_queries']}** queries")
    lines.append("")

    # Split distribution
    lines.append("## Split distribution (по primary tag)\n")
    lines.append("| Tag | Train | Test |")
    lines.append("|---|---:|---:|")
    all_tags = sorted(set(s["split"]["train_tags"]) | set(s["split"]["test_tags"]))
    for tag in all_tags:
        tr = s["split"]["train_tags"].get(tag, 0)
        te = s["split"]["test_tags"].get(tag, 0)
        lines.append(f"| {tag} | {tr} | {te} |")
    lines.append("")

    # Raw inputs
    lines.append("## Raw inputs\n")
    ri = s["raw_inputs"]
    lines.append(f"- CN passed candidates loaded: **{ri.get('cn', 0)}**")
    lines.append(f"- BM25 passed candidates loaded: **{ri.get('bm25', 0)}**")
    lines.append(f"- GP passed candidates loaded: **{ri.get('gp', 0)}**")
    lines.append("")

    # Coverage intersections
    lines.append("## Coverage по train queries (на 500)\n")
    inter = s["intersections"]
    lines.append(f"- Train qids с CN negatives: **{inter.get('train_with_cn', 0)}** "
                 f"({inter.get('pct_train_with_cn', 0)}%)")
    lines.append(f"- Train qids с BM25 negatives: **{inter.get('train_with_bm25', 0)}** "
                 f"({inter.get('pct_train_with_bm25', 0)}%)")
    lines.append(f"- Train qids с GP positives: **{inter.get('train_with_gp', 0)}** "
                 f"({inter.get('pct_train_with_gp', 0)}%)")
    lines.append(f"- Train qids с CN ∩ GP: **{inter.get('train_cn_and_gp', 0)}** "
                 f"({inter.get('pct_train_cn_and_gp', 0)}%)")
    lines.append("")

    # Conditions table
    lines.append("## Conditions\n")
    lines.append("| Condition | Records | Unique qids | qids with neg | Mean neg/rec | Mean variants/qid |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for cond in sorted(s["conditions"].keys()):
        cs = s["conditions"][cond]
        lines.append(
            f"| {cond} | {cs['n_records']} | {cs['n_unique_qids']} | "
            f"{cs['n_qids_with_any_negative']} ({cs['pct_qids_with_negatives']}%) | "
            f"{cs['mean_negatives_per_record']} | {cs['mean_variants_per_qid']} |"
        )
    lines.append("")

    # Missing qids per condition (sanity)
    lines.append("## Missing qids per condition (must be empty)\n")
    any_missing = False
    for cond, cs in s["conditions"].items():
        miss = cs.get("missing_qids") or []
        if miss:
            any_missing = True
            lines.append(f"- **{cond}**: {len(miss)} missing — first 5: {miss[:5]}")
    if not any_missing:
        lines.append("- (все condition files покрывают все train qids ✓)")
    lines.append("")

    # Sample records
    lines.append("---\n## Примеры записей (по одной из каждого condition)\n")
    for cond in sorted(s["conditions"].keys()):
        path = output_dir / f"{cond}.jsonl"
        if not path.exists():
            continue
        sample = load_jsonl(path)[:1]
        if not sample:
            continue
        r = sample[0]
        lines.append(f"### {cond}\n")
        lines.append(f"- qid: `{r['qid']}`  variant: `{r['source_variant']}`")
        lines.append(f"- query: {r['query']}")
        lines.append(f"- positive: {r['positive'][:180]}{'…' if len(r['positive']) > 180 else ''}")
        if r["negatives"]:
            lines.append(f"- negatives ({len(r['negatives'])}):")
            for n in r["negatives"][:2]:
                lines.append(f"  - {n[:180]}{'…' if len(n) > 180 else ''}")
        else:
            lines.append(f"- negatives: (none, in-batch only)")
        lines.append("")

    # Test sample
    lines.append("### test.jsonl (первая запись)\n")
    if test_set:
        r = test_set[0]
        lines.append(f"- qid: `{r['qid']}`, gold_qid: `{r['gold_qid']}`")
        lines.append(f"- query: {r['query']}")
        lines.append(f"- d⁺: {r['d_plus'][:180]}{'…' if len(r['d_plus']) > 180 else ''}")
    lines.append("")

    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

ALL_CONDITIONS = ["A", "B", "C", "D", "E"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rubq-full",
                    default=str(PROJECT_ROOT / "data/rubq/rubq_full.jsonl"))
    ap.add_argument("--cn-input",
                    default=str(PROJECT_ROOT / "outputs/cn_pipeline/prod/candidates_passed.jsonl"))
    ap.add_argument("--gp-input",
                    default=str(PROJECT_ROOT / "outputs/gp_pipeline_iter/prod/candidates_passed.jsonl"))
    ap.add_argument("--bm25-input",
                    default=str(PROJECT_ROOT / "outputs/bm25_negatives/prod/candidates_passed.jsonl"))
    ap.add_argument("--output-dir",
                    default=str(PROJECT_ROOT / "data/train"))
    ap.add_argument("--n-train", type=int, default=500)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cap-negatives", type=int, default=3,
                    help="макс. negatives на запись (default 3)")
    ap.add_argument("--conditions", nargs="+", default=ALL_CONDITIONS,
                    choices=ALL_CONDITIONS,
                    help="какие condition собирать (default: все)")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output: %s", out_dir)
    log.info("Conditions: %s", args.conditions)
    log.info("cap_negatives=%d, n_train=%d, n_test=%d, seed=%d",
             args.cap_negatives, args.n_train, args.n_test, args.seed)

    # 1. Загрузка RuBQ
    rubq = load_jsonl(Path(args.rubq_full))
    log.info("RuBQ loaded: %d пар", len(rubq))

    # 2. Split
    train_exs, test_exs, split_diag = make_train_test_split(
        rubq, n_train=args.n_train, n_test=args.n_test, seed=args.seed,
    )
    train_qids = {e["qid"] for e in train_exs}
    test_qids = {e["qid"] for e in test_exs}
    log.info("Split: train=%d, test=%d, unused=%d",
             split_diag["n_train"], split_diag["n_test"], split_diag["n_unused"])

    # Сохранить split.json
    split_data = {
        **split_diag,
        "seed": args.seed,
        "train_qids": sorted(train_qids),
        "test_qids":  sorted(test_qids),
    }
    (out_dir / "split.json").write_text(
        json.dumps(split_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 3. Загрузка negatives/positives
    cn_raw = load_jsonl(Path(args.cn_input)) if "C" in args.conditions or "E" in args.conditions else []
    bm25_raw = load_jsonl(Path(args.bm25_input)) if "B" in args.conditions else []
    gp_raw = load_jsonl(Path(args.gp_input)) if "D" in args.conditions or "E" in args.conditions else []

    log.info("Loaded: CN=%d, BM25=%d, GP=%d records",
             len(cn_raw), len(bm25_raw), len(gp_raw))

    cn_neg = group_negatives(cn_raw, _cn_text, label="CN") if cn_raw else {}
    bm25_neg = group_negatives(bm25_raw, _bm25_text, label="BM25") if bm25_raw else {}
    gp_pos = group_gp_positives(gp_raw) if gp_raw else {}

    # 4. Build conditions
    builders = {
        "A": lambda: build_A(train_exs),
        "B": lambda: build_B(train_exs, bm25_neg, cap_negatives=args.cap_negatives),
        "C": lambda: build_C(train_exs, cn_neg, cap_negatives=args.cap_negatives),
        "D": lambda: build_D(train_exs, gp_pos),
        "E": lambda: build_E(train_exs, cn_neg, gp_pos, cap_negatives=args.cap_negatives),
    }
    cond_files = {
        "A": "A_baseline.jsonl",
        "B": "B_bm25_hn.jsonl",
        "C": "C_cn.jsonl",
        "D": "D_gp.jsonl",
        "E": "E_cn_gp.jsonl",
    }

    condition_stats = {}
    for cond in args.conditions:
        records = builders[cond]()
        path = out_dir / cond_files[cond]
        save_jsonl(path, records)
        condition_stats[cond] = stats_for_condition(records, train_qids)
        log.info("[%s] %d records → %s", cond, len(records), path.name)

    # 5. Corpus + test
    corpus = build_corpus(rubq)
    save_jsonl(out_dir / "corpus.jsonl", corpus)

    test_set = build_test(test_exs, corpus)
    save_jsonl(out_dir / "test.jsonl", test_set)
    log.info("Corpus: %d docs, test: %d queries", len(corpus), len(test_set))

    # 6. Coverage intersections (только для train)
    cn_qids = set(cn_neg.keys())
    bm25_qids = set(bm25_neg.keys())
    gp_qids = set(gp_pos.keys())

    intersections = {
        "train_with_cn":   len(train_qids & cn_qids),
        "train_with_bm25": len(train_qids & bm25_qids),
        "train_with_gp":   len(train_qids & gp_qids),
        "train_cn_and_gp": len(train_qids & cn_qids & gp_qids),
        "pct_train_with_cn":   round(100 * len(train_qids & cn_qids) / max(1, len(train_qids)), 1),
        "pct_train_with_bm25": round(100 * len(train_qids & bm25_qids) / max(1, len(train_qids)), 1),
        "pct_train_with_gp":   round(100 * len(train_qids & gp_qids) / max(1, len(train_qids)), 1),
        "pct_train_cn_and_gp": round(100 * len(train_qids & cn_qids & gp_qids) / max(1, len(train_qids)), 1),
    }

    # 7. Summary + report
    raw_counts = {"cn": len(cn_raw), "bm25": len(bm25_raw), "gp": len(gp_raw)}
    summary = compute_summary(
        split_diag, corpus, test_set, condition_stats, raw_counts,
        intersections, cap_negatives=args.cap_negatives,
    )
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(summary, out_dir, train_exs, test_set)

    # ──────────────────────────────────────────────
    # stdout
    # ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TRAIN DATA BUILD — SUMMARY")
    print("=" * 70)
    print(f"Train queries:  {len(train_exs)}")
    print(f"Test queries:   {len(test_set)}")
    print(f"Corpus docs:    {len(corpus)}")
    print(f"Cap negatives:  {args.cap_negatives}")
    print()
    print("Train tag distribution:")
    for tag, n_ in sorted(split_diag["train_tags"].items(), key=lambda x: -x[1]):
        print(f"  {tag}: {n_}")
    print()
    print("Coverage по train queries:")
    print(f"  with CN negatives:   {intersections['train_with_cn']}/{len(train_qids)} "
          f"({intersections['pct_train_with_cn']}%)")
    print(f"  with BM25 negatives: {intersections['train_with_bm25']}/{len(train_qids)} "
          f"({intersections['pct_train_with_bm25']}%)")
    print(f"  with GP positives:   {intersections['train_with_gp']}/{len(train_qids)} "
          f"({intersections['pct_train_with_gp']}%)")
    print(f"  CN ∩ GP:             {intersections['train_cn_and_gp']}/{len(train_qids)} "
          f"({intersections['pct_train_cn_and_gp']}%)")
    print()
    print("Conditions:")
    print(f"  {'cond':<5} {'records':>8} {'qids':>6} {'with_neg':>9} {'mean_neg':>9} {'mean_var':>9}")
    for cond in args.conditions:
        cs = condition_stats[cond]
        print(f"  {cond:<5} {cs['n_records']:>8} {cs['n_unique_qids']:>6} "
              f"{cs['n_qids_with_any_negative']:>9} "
              f"{cs['mean_negatives_per_record']:>9} "
              f"{cs['mean_variants_per_qid']:>9}")
    print()
    print("Files:")
    print(f"  {out_dir / 'split.json'}")
    print(f"  {out_dir / 'corpus.jsonl'}")
    print(f"  {out_dir / 'test.jsonl'}")
    for cond in args.conditions:
        print(f"  {out_dir / cond_files[cond]}")
    print(f"  {out_dir / 'summary.json'}")
    print(f"  {out_dir / 'report.md'}")

    # ──────────────────────────────────────────────
    # Sanity checklist
    # ──────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("SANITY CHECKLIST")
    print("─" * 70)
    checks = [
        ("Train + test disjoint qids",
         not (train_qids & test_qids), "OK" if not (train_qids & test_qids) else "OVERLAP"),
        ("Все condition files покрывают train qids (n_unique_qids=n_train)",
         all(condition_stats[c]["n_unique_qids"] == len(train_qids)
             for c in args.conditions),
         "OK"),
        ("Test set не потерял queries (n_test_written = n_test_planned)",
         len(test_set) == args.n_test,
         f"{len(test_set)}/{args.n_test}"),
        ("Corpus содержит все test gold (gold_qid present)",
         all(r["gold_qid"] in {c["qid"] for c in corpus} for r in test_set[:50]),
         "checked first 50"),
    ]
    for name, ok, val in checks:
        mark = "✓" if ok else "✗"
        print(f"  {mark}  {name}  →  {val}")
    if all(c[1] for c in checks):
        print("\n✓ Build OK — данные готовы для 08_train_retriever.py")
    else:
        print("\n⚠ Sanity check failed. Проверь report.md.")


if __name__ == "__main__":
    main()
