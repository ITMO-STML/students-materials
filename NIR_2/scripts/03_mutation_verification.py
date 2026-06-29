#!/usr/bin/env python3
"""
Phase 0.7: Прогон Mutation Verification на контрфактах sanity-набора.

ЦЕЛЬ. Проверить, что MV даёт надёжный детерминистический сигнал «мутация
выполнена корректно». Если на 31 sanity-CF MV правильно разделяет faithful
от unfaithful мутаций, мы заменяем им DualJudge как primary truth-filter
в NegativePipeline и переходим к Фазе 2 на RuBQ-dev50.

ЧТО ВЫВОДИМ:
  outputs/sanity/mv_experiment/
    ├── mv_verdicts.jsonl        — детально по каждому CF: вердикт + диагностики
    ├── mv_summary.json          — агрегаты + cross-tab MV × judge × δ
    └── mv_report.md             — для глаза

  Консольная сводка с тремя цифрами:
    1. MV-valid rate (хотим близко к 100% на sanity — наша CN-пайплайн её делал).
    2. MV vs judge cross-tab: насколько MV ловит то, что LLM пропустила.
    3. δ-распределение внутри MV-valid (если MV-valid сосредоточены в hard,
       δ-стратификация ляжет хорошо).

ВХОД:
  --counterfactuals  outputs/sanity/counterfactuals.jsonl   (обязателен)
  --judge            outputs/sanity/judge.jsonl             (опционально, для cross-tab)
  --ce-deltas        outputs/sanity/ce_experiment/ce_deltas.jsonl  (опционально)

Запуск:
  python scripts/03_mutation_verification.py
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.mutation_verifier import MutationVerifier  # noqa: E402

log = logging.getLogger("mv_experiment")


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
# Cross-reference helpers
# ──────────────────────────────────────────────────────────────────────

def index_judge_by_doc(verdicts: list[dict]) -> dict[tuple[str, str], dict]:
    """
    Индекс judge.jsonl по (qid, doc[:200]) — потому что в judge.jsonl
    отсутствует fact_text, но есть полный doc.
    """
    idx: dict[tuple[str, str], dict] = {}
    for v in verdicts:
        if v.get("kind") != "counterfactual":
            continue
        key = (v["qid"], v["doc"][:200])
        idx[key] = v
    return idx


def index_ce_by_fact(ce_records: list[dict]) -> dict[tuple[str, str], dict]:
    """Индекс ce_deltas.jsonl по (qid, fact_text)."""
    return {(r["qid"], r["fact_text"]): r for r in ce_records}


# ──────────────────────────────────────────────────────────────────────
# Прогон MV
# ──────────────────────────────────────────────────────────────────────

def run_mv(
    cfs: list[dict],
    mv: MutationVerifier,
    judge_idx: dict[tuple[str, str], dict] | None,
    ce_idx: dict[tuple[str, str], dict] | None,
) -> list[dict]:
    out: list[dict] = []
    for cf in cfs:
        d_plus = cf["d_plus"]
        d_minus = cf["d_minus"]
        fact_text = cf["fact_text"]
        verdict = mv.verify(d_plus, d_minus, fact_text)

        # cross-ref с judge (если есть)
        judge_verdict = None
        judge_correct = None
        if judge_idx is not None:
            j = judge_idx.get((cf["qid"], d_minus[:200]))
            if j is not None:
                judge_verdict = j.get("verdict")
                judge_correct = j.get("correct")

        # cross-ref с CE δ (если есть)
        delta = None
        if ce_idx is not None:
            ce = ce_idx.get((cf["qid"], fact_text))
            if ce is not None:
                delta = ce.get("delta_ensemble")

        out.append({
            "qid": cf["qid"],
            "query": cf["query"],
            "fact_text": fact_text,
            "fact_type": cf.get("fact_type"),
            "fact_criticality": cf.get("fact_criticality"),
            "d_plus": d_plus,
            "d_minus": d_minus,
            "mv": verdict.to_dict(),
            "judge_verdict": judge_verdict,
            "judge_correct": judge_correct,
            "delta_ensemble": delta,
        })
    return out


# ──────────────────────────────────────────────────────────────────────
# Сводная статистика
# ──────────────────────────────────────────────────────────────────────

def normalize_type(t: str) -> str:
    """Грубая нормализация fact_type к категориям для группировки."""
    t = (t or "").lower()
    if "дат" in t or "год" in t or "век" in t or "месяц" in t:
        return "date"
    if "числ" in t or "колич" in t:
        return "number"
    if "имя" in t or "фамил" in t or "автор" in t or "режиссёр" in t:
        return "name"
    if "геогр" in t or "столиц" in t or "стран" in t or "город" in t \
       or "гор" in t or "мест" in t or "регион" in t:
        return "geo"
    return "other"


def compute_summary(records: list[dict]) -> dict:
    n = len(records)
    valid = [r for r in records if r["mv"]["valid"]]
    invalid = [r for r in records if not r["mv"]["valid"]]

    # Причины отбраковки
    reason_counter = Counter(r["mv"]["reason"] for r in invalid)

    # Loc/blocks статистики на valid
    if valid:
        loc = [r["mv"]["locality_ratio"] for r in valid]
        blk = [r["mv"]["n_diff_blocks"] for r in valid]
        valid_stats = {
            "n": len(valid),
            "locality_mean": round(mean(loc), 3),
            "locality_median": round(median(loc), 3),
            "locality_max": round(max(loc), 3),
            "blocks_mean": round(mean(blk), 2),
            "blocks_max": max(blk),
        }
    else:
        valid_stats = {"n": 0}

    # По типу мутации
    by_type: dict[str, dict] = defaultdict(lambda: {"valid": 0, "invalid": 0})
    for r in records:
        t = normalize_type(r.get("fact_type"))
        key = "valid" if r["mv"]["valid"] else "invalid"
        by_type[t][key] += 1

    # Cross-tab MV × judge_correct (если judge данные есть)
    has_judge = any(r["judge_correct"] is not None for r in records)
    cross_judge = None
    if has_judge:
        cross_judge = {
            "mv_valid_judge_correct":   0,
            "mv_valid_judge_wrong":     0,
            "mv_invalid_judge_correct": 0,
            "mv_invalid_judge_wrong":   0,
            "judge_correct_total":      0,
            "judge_wrong_total":        0,
        }
        for r in records:
            jc = r["judge_correct"]
            if jc is None:
                continue
            mv_valid = r["mv"]["valid"]
            if jc:
                cross_judge["judge_correct_total"] += 1
                if mv_valid:
                    cross_judge["mv_valid_judge_correct"] += 1
                else:
                    cross_judge["mv_invalid_judge_correct"] += 1
            else:
                cross_judge["judge_wrong_total"] += 1
                if mv_valid:
                    cross_judge["mv_valid_judge_wrong"] += 1
                else:
                    cross_judge["mv_invalid_judge_wrong"] += 1

    # Cross-tab MV × δ (если δ есть)
    has_delta = any(r["delta_ensemble"] is not None for r in records)
    cross_delta = None
    if has_delta:
        deltas_by_status = {"valid": [], "invalid": []}
        for r in records:
            d = r["delta_ensemble"]
            if d is None:
                continue
            key = "valid" if r["mv"]["valid"] else "invalid"
            deltas_by_status[key].append(d)
        cross_delta = {
            "valid_mean": round(mean(deltas_by_status["valid"]), 4) if deltas_by_status["valid"] else None,
            "valid_n":    len(deltas_by_status["valid"]),
            "invalid_mean": round(mean(deltas_by_status["invalid"]), 4) if deltas_by_status["invalid"] else None,
            "invalid_n":    len(deltas_by_status["invalid"]),
        }

    return {
        "n_total": n,
        "n_valid": len(valid),
        "n_invalid": len(invalid),
        "valid_rate": round(len(valid) / n, 3) if n else 0.0,
        "rejection_reasons": dict(reason_counter),
        "valid_stats": valid_stats,
        "by_type": {k: dict(v) for k, v in by_type.items()},
        "mv_vs_judge_cross_tab": cross_judge,
        "delta_in_mv_buckets": cross_delta,
    }


# ──────────────────────────────────────────────────────────────────────
# Markdown отчёт
# ──────────────────────────────────────────────────────────────────────

def write_report(records: list[dict], summary: dict, out_dir: Path) -> None:
    lines = ["# Phase 0.7 — Mutation Verification на sanity CF", ""]

    lines.append("## Сводка")
    lines.append("")
    lines.append(f"- Всего CF: **{summary['n_total']}**")
    lines.append(f"- MV-valid: **{summary['n_valid']}** ({summary['valid_rate']:.0%})")
    lines.append(f"- MV-invalid: **{summary['n_invalid']}**")
    if summary["rejection_reasons"]:
        lines.append("- Причины отбраковки:")
        for k, v in summary["rejection_reasons"].items():
            lines.append(f"  - `{k}`: {v}")
    lines.append("")

    if summary["valid_stats"].get("n", 0):
        vs = summary["valid_stats"]
        lines.append("## Локальность валидных мутаций")
        lines.append("")
        lines.append(f"- locality_ratio: mean={vs['locality_mean']}, median={vs['locality_median']}, max={vs['locality_max']}")
        lines.append(f"- n_diff_blocks:  mean={vs['blocks_mean']}, max={vs['blocks_max']}")
        lines.append("")

    if summary["by_type"]:
        lines.append("## По типу мутации")
        lines.append("")
        lines.append("| тип | valid | invalid | acc% |")
        lines.append("|---|---|---|---|")
        for t in sorted(summary["by_type"].keys()):
            d = summary["by_type"][t]
            tot = d["valid"] + d["invalid"]
            pct = d["valid"] / tot if tot else 0
            lines.append(f"| {t} | {d['valid']} | {d['invalid']} | {pct:.0%} |")
        lines.append("")

    if summary["mv_vs_judge_cross_tab"]:
        ct = summary["mv_vs_judge_cross_tab"]
        lines.append("## MV × LLM-judge cross-tab")
        lines.append("")
        lines.append("Главный вопрос: ловит ли MV то, что LLM-judge пропустила?")
        lines.append("")
        lines.append("|  | judge correct | judge wrong |")
        lines.append("|---|---|---|")
        lines.append(f"| **MV valid**   | {ct['mv_valid_judge_correct']} | {ct['mv_valid_judge_wrong']} |")
        lines.append(f"| **MV invalid** | {ct['mv_invalid_judge_correct']} | {ct['mv_invalid_judge_wrong']} |")
        lines.append("")
        # Ключевой интерпретативный момент:
        recovered = ct["mv_valid_judge_wrong"]
        lines.append(
            f"_MV восстановил **{recovered}/{ct['judge_wrong_total']}** CF, "
            f"которые LLM-judge ошибочно объявил позитивами._"
        )
        lines.append("")

    if summary["delta_in_mv_buckets"]:
        cd = summary["delta_in_mv_buckets"]
        lines.append("## δ внутри MV-buckets")
        lines.append("")
        lines.append(
            f"- MV-valid (n={cd['valid_n']}): mean δ = **{cd['valid_mean']}**"
        )
        lines.append(
            f"- MV-invalid (n={cd['invalid_n']}): mean δ = **{cd['invalid_mean']}**"
        )
        lines.append("")

    # Per-CF: что отбраковалось
    invalids = [r for r in records if not r["mv"]["valid"]]
    if invalids:
        lines.append("## Что отбраковано (полный список)")
        lines.append("")
        for r in invalids:
            lines.append(f"### `{r['qid']}` — fact `{r['fact_text']}` ({r.get('fact_type')})")
            lines.append("")
            lines.append(f"**Причина:** `{r['mv']['reason']}`")
            lines.append("")
            lines.append(f"**d⁺:** {r['d_plus'][:200]}{'…' if len(r['d_plus']) > 200 else ''}")
            lines.append("")
            lines.append(f"**d⁻:** {r['d_minus'][:200]}{'…' if len(r['d_minus']) > 200 else ''}")
            lines.append("")

    (out_dir / "mv_report.md").write_text("\n".join(lines), encoding="utf-8")
    log.info("Отчёт записан в %s", out_dir / "mv_report.md")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--counterfactuals",
        default=str(PROJECT_ROOT / "outputs/sanity/counterfactuals.jsonl"),
    )
    ap.add_argument(
        "--judge",
        default=str(PROJECT_ROOT / "outputs/sanity/judge.jsonl"),
        help="опционально — для cross-tab MV × LLM-judge",
    )
    ap.add_argument(
        "--ce-deltas",
        default=str(PROJECT_ROOT / "outputs/sanity/ce_experiment/ce_deltas.jsonl"),
        help="опционально — для cross-tab MV × δ",
    )
    ap.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs/sanity/mv_experiment"),
    )
    ap.add_argument(
        "--max-locality-ratio", type=float, default=0.30,
        help="если изменилось больше X доли символов → не локальная мутация",
    )
    ap.add_argument(
        "--max-diff-blocks", type=int, default=4,
        help="если в diff больше N блоков → слишком много правок",
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )

    cf_path = Path(args.counterfactuals)
    if not cf_path.exists():
        raise SystemExit(f"Нет файла: {cf_path}")
    cfs = load_jsonl(cf_path)
    log.info("Загружено CF: %d", len(cfs))

    judge_idx = None
    if Path(args.judge).exists():
        verdicts = load_jsonl(Path(args.judge))
        judge_idx = index_judge_by_doc(verdicts)
        log.info("Judge verdicts на CF: %d", len(judge_idx))
    else:
        log.info("Нет файла judge.jsonl — cross-tab MV × judge пропускаем")

    ce_idx = None
    if Path(args.ce_deltas).exists():
        ce_recs = load_jsonl(Path(args.ce_deltas))
        ce_idx = index_ce_by_fact(ce_recs)
        log.info("CE δ на CF: %d", len(ce_idx))
    else:
        log.info("Нет файла ce_deltas.jsonl — cross-tab MV × δ пропускаем")

    mv = MutationVerifier(
        max_locality_ratio=args.max_locality_ratio,
        max_diff_blocks=args.max_diff_blocks,
    )
    records = run_mv(cfs, mv, judge_idx, ce_idx)
    summary = compute_summary(records)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_jsonl(out_dir / "mv_verdicts.jsonl", records)
    (out_dir / "mv_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    write_report(records, summary, out_dir)

    # Консольная сводка
    print("\n" + "=" * 60)
    print("PHASE 0.7 — MUTATION VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"Всего CF:    {summary['n_total']}")
    print(f"MV-valid:    {summary['n_valid']}  ({summary['valid_rate']:.0%})")
    print(f"MV-invalid:  {summary['n_invalid']}")
    if summary["rejection_reasons"]:
        print("Причины отбраковки:")
        for k, v in summary["rejection_reasons"].items():
            print(f"  {k:35s}: {v}")
    print()
    if summary["valid_stats"].get("n", 0):
        vs = summary["valid_stats"]
        print(f"Локальность valid: locality={vs['locality_mean']:.3f} (max={vs['locality_max']:.3f}), blocks_mean={vs['blocks_mean']:.2f}")
    print()
    if summary["by_type"]:
        print(f"{'type':10s} | {'valid':>5s} | {'invalid':>7s} | acc%")
        for t in sorted(summary["by_type"].keys()):
            d = summary["by_type"][t]
            tot = d["valid"] + d["invalid"]
            print(f"{t:10s} | {d['valid']:>5d} | {d['invalid']:>7d} | {d['valid']/tot:.0%}")
    print()
    if summary["mv_vs_judge_cross_tab"]:
        ct = summary["mv_vs_judge_cross_tab"]
        print("MV × LLM-judge cross-tab:")
        print(f"  MV valid   judge correct: {ct['mv_valid_judge_correct']}")
        print(f"  MV valid   judge wrong:   {ct['mv_valid_judge_wrong']}  ← MV восстановил их")
        print(f"  MV invalid judge correct: {ct['mv_invalid_judge_correct']}")
        print(f"  MV invalid judge wrong:   {ct['mv_invalid_judge_wrong']}")
    print()
    if summary["delta_in_mv_buckets"]:
        cd = summary["delta_in_mv_buckets"]
        print(f"δ внутри MV-valid:   mean={cd['valid_mean']}  (n={cd['valid_n']})")
        print(f"δ внутри MV-invalid: mean={cd['invalid_mean']}  (n={cd['invalid_n']})")
    print()
    print(f"Подробный отчёт: {out_dir / 'mv_report.md'}")


if __name__ == "__main__":
    main()
