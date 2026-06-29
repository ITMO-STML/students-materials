#!/usr/bin/env python3
"""
Phase PoC / CN: Counterfactual Negatives — production runner.

ЧТО ЭТО.
  Для каждой пары (q, d⁺) из RuBQ:
    1. extract_facts(d⁺) — LLM выделяет атомарные verbatim-факты с criticality.
    2. mutate(q, d⁺, fact) — LLM подменяет один факт, получая d⁻.
    3. MutationVerifier — детерминистически проверяет, что подмена выполнена
       именно в позиции fact_text и локализована (не overhaul).
    4. δ = sim(q, d⁻) / sim(q, d⁺) через ансамбль e5-large + bge-m3 — диагностика.
    5. BM25 (jaccard) между d⁺ и d⁻ — фильтр от поверхностных перефраз.
    6. Запись в JSONL: все попытки в raw, прошедшие фильтры — в passed.

КЛЮЧЕВЫЕ ОТЛИЧИЯ от src/pipelines.py:NegativePipeline (устаревший).
  - БЕЗ LLM-judge. Phase 0.5 показал: CoT-judge 41.9% accuracy на CF,
    fast-judge 48%. Qwen-7B на factual tail confabulates, не справляется.
    Решение по факту мутации перенесено на детерминистический MutationVerifier.
  - БЕЗ δ-бакетов. Phase 2 показал δ сжат (median 1.0005, stdev 0.04),
    bucket-стратификация бессмысленна. δ остаётся как диагностика + hard cap
    δ ≤ 1.05 для отсева патологических случаев (d⁻ "ближе" к q чем d⁺).
  - С MutationVerifier. Концептуальный сдвиг: проверяем не "is d⁻ wrong",
    а "was the mutation carried out faithfully", что детерминируется по
    fact_text, d⁺, d⁻ без обращения к ground truth.

ФИЛЬТРЫ (passed = ALL):
  - verifier.valid           — мутация выполнена в позиции fact_text, локально.
  - δ ≤ delta_max (1.05)     — d⁻ не "ближе" к q чем d⁺ (защита от reverse).
  - bm25(d⁺, d⁻) ≤ bm25_max  — ОТКЛЮЧЁН по умолчанию (default 1.0).
                                Single-token CN мутации дают jaccard ~0.95+
                                по дизайну (меняется один токен, остальное
                                идентично). Старый порог 0.6 РЕЗАЛ валидные
                                мутации (Phase PoC.1 pilot: 0% coverage из-за
                                этого). Переоткрытие: жёсткий jaccard-фильтр
                                и CN-мутация фундаментально несовместимы —
                                это не bug, это feature метода (extra hard
                                negatives by design). Защиту от LLM-overhaul
                                делает MutationVerifier.max_locality_ratio.

ЗАПУСК.
  # Pilot (10 примеров, ~5-7 мин):
  python scripts/05_run_cn_pipeline.py --pilot

  # Production (500, ~30-60 мин):
  python scripts/05_run_cn_pipeline.py --n-examples 500

  # Тюнинг строгости:
  python scripts/05_run_cn_pipeline.py --pilot --max-facts 5 --bm25-max 0.7

ВЫХОД (в outputs/cn_pipeline/<mode>/):
  candidates_raw.jsonl        — все попытки с диагностикой
  candidates_passed.jsonl     — прошедшие все фильтры
  summary.json                — агрегированные метрики
  report.md                   — человекочитаемый отчёт с примерами

Pilot checklist печатается в stdout, аналогично 04_run_gp_pipeline.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median, stdev

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.llm_client import LLMConfig, make_client, extract_json  # noqa: E402
from src.encoders import EncoderPool  # noqa: E402
from src.filters import BM25Filter  # noqa: E402
from src.mutation_verifier import MutationVerifier  # noqa: E402
from src.prompts import (  # noqa: E402
    build_fact_extraction_messages,
    build_counterfactual_messages,
)
from src.text_utils import fuzzy_in, morph_available  # noqa: E402

log = logging.getLogger("cn_pipeline")


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
# Sampling (стратификация по primary tag, как в 04)
# ──────────────────────────────────────────────────────────────────────

def sample_examples(all_examples: list[dict], n: int, seed: int = 42) -> list[dict]:
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
# Stage 1: fact extraction
# ──────────────────────────────────────────────────────────────────────

def _classify_fact(fact_text: str, d_plus: str) -> str:
    """Тот же классификатор что в 00_sanity и 03_delta_distribution."""
    if fact_text in d_plus:
        return "verbatim"
    if fuzzy_in(fact_text, d_plus):
        return "morphological_variant"
    return "hallucinated"


def extract_facts(
    llm,
    ex: dict,
    *,
    min_criticality: int,
    max_facts: int,
) -> tuple[list[dict], dict]:
    """
    Возвращает (top_facts, diag).
    top_facts — отсортированы по criticality desc, только verbatim, len ≤ max_facts.
    diag — сколько v/morph/hallu, валидный ли JSON.
    """
    msgs = build_fact_extraction_messages(ex["query"], ex["d_plus"])
    raw = llm.generate([msgs])[0]
    parsed = extract_json(raw)
    valid_json = isinstance(parsed, list)

    verbatim, morph, hallu = [], [], []
    if valid_json:
        for item in parsed:
            if not isinstance(item, dict) or "text" not in item:
                continue
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            try:
                crit = int(item.get("criticality", 0))
            except (ValueError, TypeError):
                crit = 0
            if crit < min_criticality:
                continue
            fact = {
                "text": text,
                "type": str(item.get("type", "")).strip(),
                "criticality": crit,
                "match_status": _classify_fact(text, ex["d_plus"]),
            }
            if fact["match_status"] == "verbatim":
                verbatim.append(fact)
            elif fact["match_status"] == "morphological_variant":
                morph.append(fact)
            else:
                hallu.append(fact)

    verbatim.sort(key=lambda f: -f["criticality"])
    top = verbatim[:max_facts]
    diag = {
        "valid_json": valid_json,
        "n_verbatim": len(verbatim),
        "n_morph": len(morph),
        "n_hallu": len(hallu),
    }
    return top, diag


# ──────────────────────────────────────────────────────────────────────
# Stage 2: mutation
# ──────────────────────────────────────────────────────────────────────

def _clean_mutation_output(raw: str) -> str:
    text = raw.strip()
    # Снять обёртку из кавычек, если LLM завернула весь текст
    if len(text) > 2 and text[0] in '"«' and text[-1] in '"»':
        text = text[1:-1].strip()
    # Снять служебные префиксы (LLM иногда добавляет несмотря на инструкцию)
    prefixes = (
        "Изменённый документ:", "Изменённый текст:",
        "Документ:", "Вот изменённая версия:",
    )
    for p in prefixes:
        if text.lower().startswith(p.lower()):
            text = text[len(p):].strip()
            break
    return text


def mutate_fact(
    llm, query: str, d_plus: str, fact: dict,
) -> str:
    msgs = build_counterfactual_messages(
        query=query, d_plus=d_plus,
        fact_text=fact["text"], fact_type=fact["type"],
    )
    raw = llm.generate([msgs])[0]
    return _clean_mutation_output(raw)


# ──────────────────────────────────────────────────────────────────────
# Stage 3: per-example processing (extract → mutate → verify → δ → bm25)
# ──────────────────────────────────────────────────────────────────────

def process_example(
    llm,
    pool: EncoderPool,
    verifier: MutationVerifier,
    bm25: BM25Filter,
    ex: dict,
    *,
    min_criticality: int,
    max_facts: int,
    max_mutations_per_fact: int,
    delta_max: float,
    bm25_max: float,
) -> tuple[list[dict], dict]:
    """
    Возвращает (records, lost_reason | None).
      records: все попытки CN на этом примере (raw, с диагностикой).
      lost_reason: если ни одна попытка не прошла фильтры — сюда краткий код.
    """
    qid = ex["qid"]
    query = ex["query"]
    d_plus = ex["d_plus"]

    # 1. Extract
    top_facts, fact_diag = extract_facts(
        llm, ex,
        min_criticality=min_criticality,
        max_facts=max_facts,
    )

    if not top_facts:
        # Нет verbatim-фактов → пример теряем для CN (24% на RuBQ по Phase 2).
        return [], "no_verbatim_facts"

    records = []
    # 2-5. Mutate + verify + encode + bm25 для каждого факта
    for fact in top_facts:
        for mut_idx in range(max_mutations_per_fact):
            d_minus = mutate_fact(llm, query, d_plus, fact)

            # 3. MutationVerifier (детерминистическая проверка)
            verdict = verifier.verify(d_plus, d_minus, fact["text"])

            # 4. δ через ансамбль (всегда — это диагностика)
            try:
                delta_agg, delta_per_enc = pool.delta(query, d_plus, d_minus)
            except Exception as e:
                log.warning("[%s] encoder error: %s", qid, e)
                delta_agg = float("nan")
                delta_per_enc = {}

            # 5. BM25 (jaccard)
            bm25_ok, bm25_score = bm25.is_valid(d_plus, d_minus)

            # 6. Композитный фильтр
            reasons = []
            if not verdict.valid:
                reasons.append(f"mutation_invalid:{verdict.reason}")
            if not bm25_ok:
                reasons.append(f"bm25={bm25_score:.3f}>{bm25_max}")
            # δ может быть NaN — тогда отбрасываем
            if isinstance(delta_agg, float) and (delta_agg != delta_agg):
                reasons.append("delta_nan")
            elif delta_agg > delta_max:
                reasons.append(f"delta={delta_agg:.3f}>{delta_max}")

            passed = len(reasons) == 0

            records.append({
                "qid": qid,
                "query": query,
                "d_plus": d_plus,
                "d_minus": d_minus,
                "fact_text": fact["text"],
                "fact_type": fact["type"],
                "fact_criticality": fact["criticality"],
                "mutation_idx": mut_idx,
                # MutationVerifier
                "mutation_valid": verdict.valid,
                "mutation_reason": verdict.reason,
                "mutation_replacement": verdict.replacement,
                "mutation_locality_ratio": verdict.locality_ratio,
                "mutation_n_diff_blocks": verdict.n_diff_blocks,
                "mutation_fact_in_dminus_global": verdict.fact_in_dminus_global,
                # δ + BM25
                "delta": float(delta_agg) if delta_agg == delta_agg else None,
                "delta_per_encoder": {k: float(v) for k, v in delta_per_enc.items()},
                "bm25_jaccard": float(bm25_score),
                # Verdict
                "passed_filters": passed,
                "rejection_reason": "; ".join(reasons) if reasons else None,
                # Диагностика extract
                "extract_diag": fact_diag,
                "tags": ex.get("meta", {}).get("tags", []),
            })

    any_passed = any(r["passed_filters"] for r in records)
    lost_reason = None if any_passed else "all_candidates_rejected"
    return records, lost_reason


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
    examples: list[dict],
    lost_by_qid: dict[str, str],
    *,
    delta_max: float,
    bm25_max: float,
) -> dict:
    n_input = len(examples)
    qids_with_any = {r["qid"] for r in records}
    qids_with_passed = {r["qid"] for r in records if r["passed_filters"]}

    n_total = len(records)
    n_passed = sum(1 for r in records if r["passed_filters"])

    # Mutation verification stats
    mut_valid = [r for r in records if r["mutation_valid"]]
    mut_reasons = Counter(r["mutation_reason"] for r in records if not r["mutation_valid"])

    # δ stats (по всем records с числовой δ)
    deltas = [r["delta"] for r in records if r["delta"] is not None]
    deltas_passed = [r["delta"] for r in records if r["passed_filters"] and r["delta"] is not None]

    # BM25 stats
    bm25_all = [r["bm25_jaccard"] for r in records]
    bm25_passed = [r["bm25_jaccard"] for r in records if r["passed_filters"]]

    # Rejection breakdown (по верхним причинам — берём первую если несколько)
    rej_counter: Counter = Counter()
    for r in records:
        if r["passed_filters"]:
            continue
        rr = r["rejection_reason"] or ""
        # Категоризируем по основной причине
        if rr.startswith("mutation_invalid"):
            rej_counter["mutation_invalid"] += 1
        elif "delta_nan" in rr:
            rej_counter["delta_nan"] += 1
        elif "delta=" in rr and "bm25" not in rr:
            rej_counter["delta_too_high"] += 1
        elif "bm25=" in rr and "mutation" not in rr:
            rej_counter["bm25_too_high"] += 1
        else:
            rej_counter["multiple"] += 1

    # Lost examples breakdown
    lost_counter = Counter(lost_by_qid.values())

    # Facts diagnostics (по первой записи каждого qid)
    by_qid_first = {}
    for r in records:
        if r["qid"] not in by_qid_first:
            by_qid_first[r["qid"]] = r["extract_diag"]
    n_with_extract = len(by_qid_first)
    n_verbatim_total = sum(d["n_verbatim"] for d in by_qid_first.values())
    n_hallu_total = sum(d["n_hallu"] for d in by_qid_first.values())

    summary = {
        "config": {
            "delta_max": delta_max,
            "bm25_max": bm25_max,
        },
        "n_input_pairs": n_input,
        "coverage": {
            "n_query_with_any_candidate": len(qids_with_any),
            "n_query_with_passed_candidate": len(qids_with_passed),
            "pct_with_passed": round(100 * len(qids_with_passed) / max(1, n_input), 1),
        },
        "candidates": {
            "n_total": n_total,
            "n_passed": n_passed,
            "pass_rate": round(n_passed / max(1, n_total), 4),
        },
        "mutation_verification": {
            "n_valid": len(mut_valid),
            "valid_rate": round(len(mut_valid) / max(1, n_total), 4),
            "invalid_reasons": dict(mut_reasons),
        },
        "rejection_breakdown": dict(rej_counter),
        "examples_lost": dict(lost_counter),
        "delta_overall": {
            "n": len(deltas),
            "mean":   round(mean(deltas), 4) if deltas else None,
            "median": round(median(deltas), 4) if deltas else None,
            "stdev":  round(stdev(deltas), 4) if len(deltas) > 1 else None,
            "min":    round(min(deltas), 4) if deltas else None,
            "max":    round(max(deltas), 4) if deltas else None,
            **_percentiles(deltas),
        },
        "delta_passed_only": {
            "n": len(deltas_passed),
            "mean":   round(mean(deltas_passed), 4) if deltas_passed else None,
            "median": round(median(deltas_passed), 4) if deltas_passed else None,
        },
        "bm25_overall": {
            "n": len(bm25_all),
            "mean":   round(mean(bm25_all), 4) if bm25_all else None,
            "median": round(median(bm25_all), 4) if bm25_all else None,
        },
        "bm25_passed_only": {
            "n": len(bm25_passed),
            "mean":   round(mean(bm25_passed), 4) if bm25_passed else None,
            "median": round(median(bm25_passed), 4) if bm25_passed else None,
        },
        "facts_diagnostic": {
            "n_examples_with_extract": n_with_extract,
            "n_verbatim_total": n_verbatim_total,
            "n_hallucinated_total": n_hallu_total,
            "mean_verbatim_per_doc": round(
                n_verbatim_total / max(1, n_with_extract), 2
            ),
        },
    }
    return summary


# ──────────────────────────────────────────────────────────────────────
# Report.md (sddsdadsadasdsadsdadasdasdadsaasdsadsadasdadsadasd)
# ──────────────────────────────────────────────────────────────────────

def write_report(
    summary: dict,
    records: list[dict],
    examples: list[dict],
    lost_by_qid: dict[str, str],
    output_dir: Path,
    mode_name: str,
) -> None:
    cfg = summary["config"]
    cov = summary["coverage"]
    cand = summary["candidates"]
    mv = summary["mutation_verification"]
    do = summary["delta_overall"]
    dop = summary["delta_passed_only"]
    bm25 = summary["bm25_overall"]
    bm25p = summary["bm25_passed_only"]
    fd = summary["facts_diagnostic"]

    lines = []
    lines.append(f"# CN Pipeline — {mode_name}\n")
    lines.append(f"- Input pairs: **{summary['n_input_pairs']}**")
    lines.append(f"- Coverage (≥1 passed):     "
                 f"**{cov['n_query_with_passed_candidate']}/{summary['n_input_pairs']}** "
                 f"({cov['pct_with_passed']}%)")
    lines.append(f"- Total candidates:         {cand['n_total']}, "
                 f"passed: **{cand['n_passed']}** ({cand['pass_rate']:.1%})")
    lines.append(f"- Filters: bm25 ≤ {cfg['bm25_max']}, δ ≤ {cfg['delta_max']}, "
                 f"MutationVerifier.valid")
    lines.append("")

    # Mutation verification
    lines.append("## Mutation verification\n")
    lines.append(f"- Valid mutations: **{mv['n_valid']}/{cand['n_total']}** "
                 f"({mv['valid_rate']:.1%})")
    if mv["invalid_reasons"]:
        lines.append("- Invalid reasons:")
        for reason, n in sorted(mv["invalid_reasons"].items(), key=lambda x: -x[1]):
            lines.append(f"  - `{reason}`: {n}")
    lines.append("")

    # Rejection breakdown
    lines.append("## Rejection breakdown\n")
    rb = summary["rejection_breakdown"]
    if rb:
        for cause, n in sorted(rb.items(), key=lambda x: -x[1]):
            lines.append(f"- `{cause}`: {n}")
    else:
        lines.append("- (нет отказов)")
    lines.append("")

    # Examples lost
    lines.append("## Examples lost\n")
    el = summary["examples_lost"]
    if el:
        for reason, n in sorted(el.items(), key=lambda x: -x[1]):
            lines.append(f"- `{reason}`: {n}")
    else:
        lines.append("- (нет потерянных)")
    lines.append("")

    # δ stats
    lines.append("## δ distribution\n")
    if do["n"]:
        stdev_str = f"{do['stdev']:.4f}" if do["stdev"] is not None else "n/a"
        lines.append(f"**All candidates (n={do['n']}):**")
        lines.append(f"- mean={do['mean']:.4f}  median={do['median']:.4f}  "
                     f"stdev={stdev_str}  "
                     f"range=[{do['min']:.4f}, {do['max']:.4f}]")
        lines.append(f"- quantiles: p10={do['p10']}  p25={do['p25']}  "
                     f"p50={do['p50']}  p75={do['p75']}  p90={do['p90']}")
        if dop["n"]:
            lines.append(f"\n**Passed only (n={dop['n']}):**")
            lines.append(f"- mean={dop['mean']:.4f}  median={dop['median']:.4f}")
    else:
        lines.append("(нет данных)")
    lines.append("")

    # BM25 stats
    lines.append("## BM25 (jaccard) distribution\n")
    if bm25["n"]:
        lines.append(f"- All: mean={bm25['mean']:.4f}  median={bm25['median']:.4f}")
        if bm25p["n"]:
            lines.append(f"- Passed: mean={bm25p['mean']:.4f}  "
                         f"median={bm25p['median']:.4f}")
    lines.append("")

    # Facts stats
    lines.append("## Facts extraction\n")
    lines.append(f"- Examples с extract: {fd['n_examples_with_extract']}")
    lines.append(f"- Verbatim facts total: {fd['n_verbatim_total']} "
                 f"(mean {fd['mean_verbatim_per_doc']} per doc)")
    lines.append(f"- Hallucinated facts: {fd['n_hallucinated_total']}")
    lines.append("")

    # Примеры мутаций (3 successful + 2 failed)
    lines.append("---\n## Примеры мутаций\n")
    qd_map = {ex["qid"]: ex for ex in examples}

    passed_records = [r for r in records if r["passed_filters"]]
    failed_records = [r for r in records if not r["passed_filters"]]

    for label, items, take in [
        ("Прошедшие фильтры", passed_records, 3),
        ("Отклонённые", failed_records, 2),
    ]:
        lines.append(f"### {label}\n")
        seen_qids = set()
        shown = 0
        for r in items:
            if r["qid"] in seen_qids:
                continue
            seen_qids.add(r["qid"])
            lines.append(f"**{r['qid']}** — fact: `{r['fact_text']}` "
                         f"({r['fact_type']}, criticality={r['fact_criticality']})\n")
            lines.append(f"- Query: {r['query']}")
            dp = r["d_plus"][:220].replace("\n", " ")
            dm = r["d_minus"][:220].replace("\n", " ")
            lines.append(f"- d⁺: {dp}{'…' if len(r['d_plus']) > 220 else ''}")
            lines.append(f"- d⁻: {dm}{'…' if len(r['d_minus']) > 220 else ''}")
            lines.append(f"- Replacement: `{r['mutation_replacement']}`")
            d_str = f"{r['delta']:.3f}" if r["delta"] is not None else "—"
            lines.append(f"- δ={d_str}, bm25={r['bm25_jaccard']:.3f}, "
                         f"locality={r['mutation_locality_ratio']:.3f}")
            if not r["passed_filters"]:
                lines.append(f"- ✗ Reason: {r['rejection_reason']}")
            lines.append("")
            shown += 1
            if shown >= take:
                break

    # Lost qids
    if lost_by_qid:
        lines.append("### Потерянные примеры\n")
        for qid, reason in list(lost_by_qid.items())[:5]:
            ex = qd_map.get(qid)
            if ex:
                q = ex["query"][:100]
                lines.append(f"- **{qid}** (`{reason}`): {q}")
        lines.append("")

    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs/default.yaml"))
    ap.add_argument("--input",  default=str(PROJECT_ROOT / "data/rubq/rubq_full.jsonl"))
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--n-examples", type=int, default=500)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--max-facts", type=int, default=3,
                    help="сколько top-фактов на пример (default: 3)")
    ap.add_argument("--min-criticality", type=int, default=3)
    ap.add_argument("--max-mutations-per-fact", type=int, default=1,
                    help="N CF на факт (default: 1, больше = шире покрытие)")
    ap.add_argument("--delta-max", type=float, default=1.05,
                    help="δ > этого → отбрасываем (d⁻ ближе к q чем d⁺)")
    ap.add_argument("--bm25-max", type=float, default=1.0,
                    help="jaccard > этого → d⁻ слишком похож на d⁺. "
                         "DEFAULT 1.0 = filter ОТКЛЮЧЁН (accept all). "
                         "При single-token CN мутации jaccard ВСЕГДА ~0.95+, "
                         "и старый порог 0.6 резал валидные мутации. "
                         "MutationVerifier (max_locality_ratio) уже защищает "
                         "от overhaul. Используй 0.6 только если хочешь принудительно "
                         "отсекать paraphrase-like CFs (не нужно для текущего pipeline).")
    ap.add_argument("--mv-max-locality", type=float, default=0.30,
                    help="MutationVerifier: верх граница на долю изменённых символов")
    ap.add_argument("--mv-max-blocks", type=int, default=12,
                    help="MutationVerifier: верх граница на число diff-блоков")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    mode_name = "pilot" if args.pilot else "prod"
    n = 10 if args.pilot else args.n_examples
    out_dir = Path(args.output_dir) if args.output_dir else \
              (PROJECT_ROOT / "outputs" / "cn_pipeline" / mode_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Режим: %s, output: %s", mode_name, out_dir)
    log.info("max_facts=%d  min_criticality=%d  max_mut_per_fact=%d",
             args.max_facts, args.min_criticality, args.max_mutations_per_fact)
    log.info("Filters: bm25 ≤ %.2f, δ ≤ %.2f, MutationVerifier(locality ≤ %.2f, blocks ≤ %d)",
             args.bm25_max, args.delta_max, args.mv_max_locality, args.mv_max_blocks)
    log.info("pymorphy3: %s", "доступен" if morph_available() else "НЕТ")

    examples = load_jsonl(Path(args.input))
    log.info("Загружено пар: %d", len(examples))
    examples = sample_examples(examples, n, seed=args.seed)
    log.info("К обработке: %d", len(examples))

    # Компоненты
    llm_cfg = LLMConfig.from_dict(cfg["llm"])
    log.info("LLM: %s (backend=%s)", llm_cfg.model_name, llm_cfg.backend)
    llm = make_client(llm_cfg)

    log.info("Загружаю encoders...")
    pool = EncoderPool.from_config(cfg["encoders"])

    verifier = MutationVerifier(
        max_locality_ratio=args.mv_max_locality,
        max_diff_blocks=args.mv_max_blocks,
    )
    bm25 = BM25Filter(similarity_max=args.bm25_max)

    # Цикл
    try:
        from tqdm import tqdm
        pbar = tqdm(examples, desc="CN", unit="ex")
    except ImportError:
        pbar = examples

    all_records: list[dict] = []
    lost_by_qid: dict[str, str] = {}
    for ex in pbar:
        recs, lost = process_example(
            llm, pool, verifier, bm25, ex,
            min_criticality=args.min_criticality,
            max_facts=args.max_facts,
            max_mutations_per_fact=args.max_mutations_per_fact,
            delta_max=args.delta_max,
            bm25_max=args.bm25_max,
        )
        all_records.extend(recs)
        if lost:
            lost_by_qid[ex["qid"]] = lost
        if args.pilot:
            n_pass = sum(1 for r in recs if r["passed_filters"])
            n_valid = sum(1 for r in recs if r["mutation_valid"])
            log.info("[%s] facts→cands=%d  valid_mut=%d  passed=%d  lost=%s",
                     ex["qid"], len(recs), n_valid, n_pass, lost or "—")

    # Запись
    save_jsonl(out_dir / "candidates_raw.jsonl", all_records)
    save_jsonl(out_dir / "candidates_passed.jsonl",
               [r for r in all_records if r["passed_filters"]])

    summary = compute_summary(
        all_records, examples, lost_by_qid,
        delta_max=args.delta_max, bm25_max=args.bm25_max,
    )
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(summary, all_records, examples, lost_by_qid, out_dir, mode_name)

    # ──────────────────────────────────────────────
    # Печать сводки
    # ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"CN PIPELINE — {mode_name.upper()}")
    print("=" * 70)
    cov = summary["coverage"]
    cand = summary["candidates"]
    mv = summary["mutation_verification"]
    do = summary["delta_overall"]
    fd = summary["facts_diagnostic"]

    print(f"Input pairs:           {summary['n_input_pairs']}")
    print(f"Coverage (≥1 passed):  {cov['n_query_with_passed_candidate']}/"
          f"{summary['n_input_pairs']} ({cov['pct_with_passed']}%)")
    print(f"Candidates total:      {cand['n_total']}, "
          f"passed: {cand['n_passed']} ({cand['pass_rate']:.1%})")
    print(f"Mutation valid:        {mv['n_valid']}/{cand['n_total']} "
          f"({mv['valid_rate']:.1%})")
    print(f"Verbatim facts/doc:    {fd['mean_verbatim_per_doc']}")
    print()
    if do["n"]:
        stdev_str = f"{do['stdev']:.4f}" if do["stdev"] is not None else "n/a"
        print(f"δ overall (n={do['n']}):")
        print(f"  mean={do['mean']:.4f}  median={do['median']:.4f}  "
              f"stdev={stdev_str}")
        print(f"  range=[{do['min']:.4f}, {do['max']:.4f}]  "
              f"p10={do['p10']}  p90={do['p90']}")
    print()
    if summary["rejection_breakdown"]:
        print("Rejection breakdown:")
        for cause, n_ in sorted(summary["rejection_breakdown"].items(), key=lambda x: -x[1]):
            print(f"  {cause}: {n_}")
    print()
    if summary["examples_lost"]:
        print("Examples lost:")
        for reason, n_ in sorted(summary["examples_lost"].items(), key=lambda x: -x[1]):
            print(f"  {reason}: {n_}")
    print()
    print("Files:")
    print(f"  {out_dir / 'candidates_raw.jsonl'}")
    print(f"  {out_dir / 'candidates_passed.jsonl'}")
    print(f"  {out_dir / 'summary.json'}")
    print(f"  {out_dir / 'report.md'}")

    if args.pilot:
        print("\n" + "─" * 70)
        print("PILOT CHECKLIST (CN)")
        print("─" * 70)
        # Ориентир: с учётом 24% lost (Phase 2) — coverage около 75% это потолок данных.
        # mutation valid_rate ~85-95% на калибровке MutationVerifier (Phase 0.7).
        # δ должен быть около 1.0 (Phase 2 median=1.0005), не уезжать сильно вверх.
        checks = [
            ("Coverage ≥70% (учёт ~24% lost no_verbatim в Phase 2)",
             cov["pct_with_passed"] >= 70, f"{cov['pct_with_passed']}%"),
            ("Mutation valid_rate ≥80%",
             mv["valid_rate"] >= 0.80, f"{mv['valid_rate']:.1%}"),
            ("δ median ≤ 1.05 (нет систематического reverse)",
             do["median"] is not None and do["median"] <= 1.05,
             f"{do['median']}"),
            ("BM25 passed mean ≥ 0.85 (single-token CFs дают high jaccard by design)",
             summary["bm25_passed_only"]["mean"] is not None
             and summary["bm25_passed_only"]["mean"] >= 0.85,
             f"{summary['bm25_passed_only']['mean']}"),
            ("Verbatim facts/doc ≥ 1.0",
             fd["mean_verbatim_per_doc"] >= 1.0,
             f"{fd['mean_verbatim_per_doc']}"),
        ]
        for name, ok, val in checks:
            mark = "✓" if ok else "✗"
            print(f"  {mark}  {name}  →  {val}")
        if all(c[1] for c in checks):
            print("\n✓ Pilot OK — запускай production: --n-examples 500 (без --pilot)")
        else:
            print("\n⚠ Есть провалы. Смотри report.md → Rejection breakdown / Examples lost.")
            print("  Возможные причины:")
            print("    - mutation_invalid ↑ → ослабь --mv-max-locality (0.30 → 0.40)")
            print("      или --mv-max-blocks (12 → 20).")
            print("    - bm25_too_high ↑   → НЕ ожидаемо при default bm25_max=1.0;")
            print("      если ты понизил порог вручную и режет — верни 1.0 (CFs дают jaccard ~0.95+ by design).")
            print("    - no_verbatim_facts → промпт extract выдаёт пересказы, не подстроки.")
            print("    - delta_too_high   → CF систематически 'ближе' к q; нужен анализ.")


if __name__ == "__main__":
    main()
