#!/usr/bin/env python3
"""
Phase PoC / GP: Gradient Positives — production runner (ITERATIVE v2).

КЛЮЧЕВОЕ ИЗМЕНЕНИЕ от v1:

  v1 (single-pass, ОТКАЗАЛИСЬ):  один промпт просит LLM сгенерировать N версий
    разных уровней (near/mid/far) за один вызов. Pilot показал три красных флага:
    length mean = 0.56 (LLM сжимает текст вдвое), distinct levels = 0/10
    (схлопывается в один режим), LLM-judge yes-rate = 48% (сжатые версии теряют
    факты). Причина — LLM не справляется с задачей "сгенерируй заведомо разные
    уровни" одним батчем.

  v2 (iterative, ЭТОТ скрипт):  natural drift через цепочку независимых
    переписываний. d_0 = d⁺, d_k = paraphrase(d_{k-1}). На каждой итерации
    sim(d⁺, d_k) меряется, и judge проверяет — отвечает ли d_k всё ещё на q.
    Когда judge говорит "нет" — chain обрывается, последние валидные d_k идут
    в positives. Длина контролируется явным constraint в промпте.

ВАЖНО.

  - Targets из configs/default.yaml (positives.similarity_targets) больше НЕ
    используются. Сохраняем continuous sim(d⁺, d_k) и индекс итерации; выбор
    positives по уровню сложности — задача training-time data builder'а.

  - LLM-judge меняет роль: был gate-keeper всех кандидатов, стал stopping
    criterion цепочки. Это решает проблему "ложных отказов" на хорошо
    переформулированных версиях.

  - Для разнообразия — несколько независимых chain на одну query (--n-chains).
    Внутри chain — итеративный drift. На выходе хранятся все d_k всех chain.

ЗАПУСК.

  # Pilot (10 примеров, ~5-10 мин):
  python scripts/04_run_gp_pipeline.py --pilot

  # Production:
  python scripts/04_run_gp_pipeline.py --n-examples 500

  # Без судьи (для быстрой отладки, цепочки идут до --n-iterations):
  python scripts/04_run_gp_pipeline.py --pilot --skip-judge

  # Глубже / шире:
  python scripts/04_run_gp_pipeline.py --pilot --n-iterations 6 --n-chains 3
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.llm_client import LLMConfig, make_client  # noqa: E402
from src.encoders import EncoderPool  # noqa: E402


log = logging.getLogger("gp_iter")


# ──────────────────────────────────────────────────────────────────────
# Промпт для одной итерации перефразирования
# ──────────────────────────────────────────────────────────────────────
#
# Этот промпт намеренно простой — одна задача за один вызов.
# Главные constraint-ы: сохранить ключевые факты + сохранить длину.
# Без этих constraint-ов модель скатывается к "сделай короче"
# (что и убило single-pass v1).

PARAPHRASE_SYSTEM = (
    "Ты — редактор, который переписывает текст другими словами, сохраняя его смысл "
    "и фактическое содержание."
)

PARAPHRASE_USER_TEMPLATE = """Запрос: {query}

Документ, отвечающий на запрос:
{doc}

Перепиши этот документ другими словами:
- Сохрани ВСЕ факты, нужные для ответа на запрос (имена, даты, числа, места).
- Сохрани общую длину текста (±20% от оригинала).
- Используй другую лексику и порядок изложения, но не меняй смысл.
- Не добавляй информацию, которой нет в оригинале.

Верни только переписанный документ, без преамбулы и пояснений."""


def build_paraphrase_messages(query: str, doc: str) -> list[dict]:
    return [
        {"role": "system", "content": PARAPHRASE_SYSTEM},
        {"role": "user",   "content": PARAPHRASE_USER_TEMPLATE.format(
            query=query, doc=doc,
        )},
    ]


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
# Sampling
# ──────────────────────────────────────────────────────────────────────

def sample_examples(all_examples: list[dict], n: int, seed: int = 42) -> list[dict]:
    """Стратифицированный сэмпл по primary tag (чтобы не получить только 1-hop)."""
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
# Iterative chain rewriting
# ──────────────────────────────────────────────────────────────────────

def _clean_llm_output(raw: str) -> str:
    """LLM иногда заворачивает в кавычки или добавляет преамбулу."""
    text = raw.strip()
    prefixes = (
        "Переписанный документ:", "Переписанный текст:", "Документ:",
        "Текст:", "Вот переписанный документ:", "Вот переписанный текст:",
    )
    for p in prefixes:
        if text.lower().startswith(p.lower()):
            text = text[len(p):].strip()
            break
    if len(text) > 2 and text[0] in '"«' and text[-1] in '"»':
        text = text[1:-1].strip()
    return text


def generate_chain(
    llm,
    pool: EncoderPool,
    llm_judge,
    query: str,
    d_plus: str,
    n_iterations: int,
    chain_idx: int,
    stop_on_judge_no: bool = True,
) -> list[dict]:
    """
    Одна итеративная цепочка переписываний.

    Возвращает записи по итерациям 1..K (запись для k=0 = оригинал тривиальна,
    не возвращаем). Если judge говорит "нет" на итерации k — последняя запись
    помечается stop_reason="judge_no" и цикл прерывается.
    """
    chain = []
    current = d_plus

    for k in range(1, n_iterations + 1):
        msgs = build_paraphrase_messages(query, current)
        raw = llm.generate([msgs])[0]
        new_text = _clean_llm_output(raw)

        if not new_text:
            chain.append({
                "iteration": k,
                "text": "",
                "sim_to_d_plus": None,
                "length_ratio": 0.0,
                "llm_verdict": None,
                "chain_idx": chain_idx,
                "stop_reason": "empty_output",
            })
            break

        if new_text == current:
            chain.append({
                "iteration": k,
                "text": new_text,
                "sim_to_d_plus": None,
                "length_ratio": len(new_text) / max(1, len(d_plus)),
                "llm_verdict": None,
                "chain_idx": chain_idx,
                "stop_reason": "no_change_from_prev",
            })
            break

        # sim к ОРИГИНАЛУ d_plus — это позволяет видеть общий drift, а не только step-to-step
        sims_agg, sims_per_enc = pool.similarity_batch(d_plus, [new_text], role="dd")
        sim_to_dp = float(sims_agg[0])
        per_enc = {kk: float(vv[0]) for kk, vv in sims_per_enc.items()}

        length_ratio = len(new_text) / max(1, len(d_plus))

        verdict = None
        if llm_judge is not None:
            try:
                verdict = llm_judge.judge(query, new_text)
            except Exception as e:
                log.warning("LLM-judge error at iter=%d: %s", k, e)
                verdict = "error"

        record = {
            "iteration": k,
            "text": new_text,
            "sim_to_d_plus": sim_to_dp,
            "sim_per_enc": per_enc,
            "length_ratio": round(length_ratio, 3),
            "llm_verdict": verdict,
            "chain_idx": chain_idx,
            "stop_reason": None,
        }

        should_stop = False
        if stop_on_judge_no and verdict == "no":
            record["stop_reason"] = "judge_no"
            should_stop = True
        elif length_ratio < 0.4 or length_ratio > 2.5:
            record["stop_reason"] = "length_blowout"
            should_stop = True

        chain.append(record)
        if should_stop:
            break

        current = new_text  # следующая итерация перефразирует уже d_k

    return chain


def process_example(
    llm, pool, llm_judge, ex: dict, n_chains: int, n_iterations: int,
) -> list[dict]:
    """Запускает n_chains независимых цепочек на одну query."""
    all_records = []
    for chain_idx in range(n_chains):
        chain = generate_chain(
            llm, pool, llm_judge,
            query=ex["query"], d_plus=ex["d_plus"],
            n_iterations=n_iterations,
            chain_idx=chain_idx,
        )
        for r in chain:
            r["qid"] = ex["qid"]
            r["query"] = ex["query"]
            all_records.append(r)
    return all_records


# ──────────────────────────────────────────────────────────────────────
# Pass/fail criterion
# ──────────────────────────────────────────────────────────────────────

def is_passed(rec: dict) -> bool:
    if not rec.get("text"):
        return False
    if rec.get("sim_to_d_plus") is None:
        return False
    if rec.get("llm_verdict") == "no":
        return False
    if rec.get("stop_reason") in (
        "judge_no", "empty_output", "length_blowout", "no_change_from_prev"
    ):
        return False
    lr = rec.get("length_ratio", 1.0)
    if not (0.5 <= lr <= 1.8):
        return False
    return True


# ──────────────────────────────────────────────────────────────────────
# Метрики
# ──────────────────────────────────────────────────────────────────────

def quartile(xs):
    if not xs:
        return None
    s = sorted(xs)
    return {
        "n": len(s),
        "mean": round(mean(s), 4),
        "median": round(median(s), 4),
        "min": round(min(s), 4),
        "max": round(max(s), 4),
        "p25": round(s[len(s) // 4], 4) if len(s) >= 4 else None,
        "p75": round(s[3 * len(s) // 4], 4) if len(s) >= 4 else None,
    }


def compute_summary(records: list[dict], examples: list[dict]) -> dict:
    n_examples = len(examples)

    by_qid = defaultdict(list)
    for r in records:
        by_qid[r["qid"]].append(r)

    n_with_any = sum(1 for qid in by_qid if by_qid[qid])
    n_with_passed = sum(1 for lst in by_qid.values() if any(is_passed(r) for r in lst))

    # Chain depths
    chain_depths = defaultdict(list)
    chain_stop_reasons = defaultdict(int)
    for r in records:
        key = (r["qid"], r.get("chain_idx", 0))
        chain_depths[key].append(r["iteration"])
        if r.get("stop_reason"):
            chain_stop_reasons[r["stop_reason"]] += 1
    max_depth_per_chain = [max(v) for v in chain_depths.values()]

    # Sim/length/judge по индексу итерации
    sim_by_iter = defaultdict(list)
    length_by_iter = defaultdict(list)
    yes_by_iter = defaultdict(list)
    for r in records:
        if r.get("sim_to_d_plus") is not None:
            sim_by_iter[r["iteration"]].append(r["sim_to_d_plus"])
            length_by_iter[r["iteration"]].append(r["length_ratio"])
            if r.get("llm_verdict") is not None:
                yes_by_iter[r["iteration"]].append(1 if r["llm_verdict"] == "yes" else 0)

    per_iter_sim_stats = {k: quartile(v) for k, v in sorted(sim_by_iter.items())}
    per_iter_len_stats = {k: quartile(v) for k, v in sorted(length_by_iter.items())}
    per_iter_judge_yes = {k: round(sum(v) / len(v), 3) if v else None
                          for k, v in sorted(yes_by_iter.items())}

    # Bucket counts (только passed)
    bucket_counts = {
        "near (0.85-1.00)": 0, "mid (0.70-0.85)": 0,
        "far (0.50-0.70)": 0,  "out (<0.50)": 0,
    }
    for r in records:
        if not is_passed(r):
            continue
        s = r["sim_to_d_plus"]
        if s >= 0.85:  bucket_counts["near (0.85-1.00)"] += 1
        elif s >= 0.70: bucket_counts["mid (0.70-0.85)"] += 1
        elif s >= 0.50: bucket_counts["far (0.50-0.70)"] += 1
        else:           bucket_counts["out (<0.50)"] += 1

    # Distinct levels per query
    distinct_levels_per_query = []
    for lst in by_qid.values():
        levels = set()
        for r in lst:
            if not is_passed(r):
                continue
            s = r["sim_to_d_plus"]
            if s >= 0.85:   levels.add("near")
            elif s >= 0.70: levels.add("mid")
            elif s >= 0.50: levels.add("far")
        distinct_levels_per_query.append(len(levels))
    n_query_3 = sum(1 for x in distinct_levels_per_query if x >= 3)
    n_query_2 = sum(1 for x in distinct_levels_per_query if x >= 2)

    return {
        "n_input_pairs": n_examples,
        "coverage": {
            "n_query_with_any_candidate":    n_with_any,
            "n_query_with_passed_candidate": n_with_passed,
            "n_query_with_2plus_levels":     n_query_2,
            "n_query_with_3plus_levels":     n_query_3,
            "pct_with_passed": round(n_with_passed / max(1, n_examples) * 100, 1),
        },
        "candidates": {
            "n_total":  len(records),
            "n_passed": sum(1 for r in records if is_passed(r)),
        },
        "chain_depth": {
            "mean_max_depth":   round(mean(max_depth_per_chain), 2) if max_depth_per_chain else 0,
            "median_max_depth": median(max_depth_per_chain)        if max_depth_per_chain else 0,
        },
        "stop_reasons": dict(chain_stop_reasons),
        "sim_by_iteration":       per_iter_sim_stats,
        "length_by_iteration":    per_iter_len_stats,
        "judge_yes_by_iteration": per_iter_judge_yes,
        "bucket_counts_passed":   bucket_counts,
    }


# ──────────────────────────────────────────────────────────────────────
# Markdown report
# ──────────────────────────────────────────────────────────────────────

def write_report(summary, records, examples, output_dir: Path, mode_name: str):
    lines = [f"# GP Pipeline (Iterative) — {mode_name}\n"]
    cov  = summary["coverage"]
    cand = summary["candidates"]
    lines.append(f"- Input pairs: **{summary['n_input_pairs']}**")
    lines.append(f"- Coverage (≥1 passed):     **{cov['n_query_with_passed_candidate']}/{summary['n_input_pairs']}** ({cov['pct_with_passed']}%)")
    lines.append(f"- Query w/ ≥2 sim-levels:   {cov['n_query_with_2plus_levels']}")
    lines.append(f"- Query w/ ≥3 sim-levels:   {cov['n_query_with_3plus_levels']}")
    lines.append(f"- Total candidates:         {cand['n_total']}, passed: **{cand['n_passed']}**")
    lines.append("")
    d = summary["chain_depth"]
    lines.append(f"## Chain depth")
    lines.append(f"- Mean max iteration reached: **{d['mean_max_depth']}**")
    lines.append(f"- Median: {d['median_max_depth']}")
    if summary["stop_reasons"]:
        lines.append(f"- Stop reasons:")
        for r, n in sorted(summary["stop_reasons"].items(), key=lambda x: -x[1]):
            lines.append(f"  - `{r}`: {n}")
    lines.append("")
    lines.append("## Drift по итерациям")
    lines.append("")
    lines.append("| iter | n  | sim mean  | sim median | sim min  | length mean | judge yes-rate |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for k in sorted(summary["sim_by_iteration"].keys()):
        s = summary["sim_by_iteration"][k]
        l = summary["length_by_iteration"].get(k, {})
        y = summary["judge_yes_by_iteration"].get(k)
        y_str = f"{y:.1%}" if y is not None else "—"
        lines.append(f"| {k} | {s['n']} | {s['mean']:.3f} | {s['median']:.3f} | {s['min']:.3f} | {l.get('mean', 0):.2f} | {y_str} |")
    lines.append("")
    lines.append("**Что хотим увидеть:**")
    lines.append("- `sim mean` монотонно убывает с ростом `iter` → есть natural gradient")
    lines.append("- `length mean` остаётся в [0.7, 1.4] → текст не сжимается")
    lines.append("- `judge yes-rate` высокий на k=1-2, падает на k=4+ → judge корректно ловит точку срыва")
    lines.append("")
    lines.append("## Distribution passed по диапазонам sim")
    lines.append("")
    for bucket, n in summary["bucket_counts_passed"].items():
        lines.append(f"- `{bucket}`: {n}")
    lines.append("")
    lines.append("---\n## Примеры цепочек\n")
    qd_map = {ex["qid"]: ex for ex in examples}
    by_qid = defaultdict(list)
    for r in records:
        by_qid[r["qid"]].append(r)
    for qid in list(by_qid.keys())[:3]:
        ex = qd_map.get(qid)
        if not ex: continue
        lines.append(f"### {qid}\n")
        lines.append(f"**Query:** {ex['query']}\n")
        lines.append(f"**d⁺ (iter 0):** {ex['d_plus'][:280]}{'…' if len(ex['d_plus']) > 280 else ''}\n")
        by_chain = defaultdict(list)
        for r in by_qid[qid]:
            by_chain[r.get("chain_idx", 0)].append(r)
        for cidx in sorted(by_chain.keys()):
            lines.append(f"**Chain {cidx}:**\n")
            for r in sorted(by_chain[cidx], key=lambda x: x["iteration"]):
                mark = "✓" if is_passed(r) else "✗"
                stop = f"  STOP={r['stop_reason']}" if r.get("stop_reason") else ""
                sim = r["sim_to_d_plus"]
                sim_str = f"{sim:.3f}" if sim is not None else "—"
                lr = r.get("length_ratio", 0)
                v = r.get("llm_verdict", "?")
                lines.append(f"- {mark} **iter {r['iteration']}** sim={sim_str} len={lr:.2f} judge={v}{stop}")
                if r["text"]:
                    txt = r["text"][:220].replace("\n", " ")
                    lines.append(f"  > {txt}{'…' if len(r['text']) > 220 else ''}")
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
    ap.add_argument("--n-chains",     type=int, default=2,
                    help="независимых цепочек на пример (для разнообразия)")
    ap.add_argument("--n-iterations", type=int, default=4,
                    help="максимум итераций в каждой цепочке")
    ap.add_argument("--skip-judge", action="store_true")
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
              (PROJECT_ROOT / "outputs" / "gp_pipeline_iter" / mode_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Режим: %s, output: %s", mode_name, out_dir)
    log.info("Chains per query: %d, max iterations: %d", args.n_chains, args.n_iterations)

    examples = load_jsonl(Path(args.input))
    log.info("Загружено пар: %d", len(examples))
    examples = sample_examples(examples, n, seed=args.seed)
    log.info("К обработке: %d", len(examples))

    llm_cfg = LLMConfig.from_dict(cfg["llm"])
    log.info("LLM: %s (backend=%s)", llm_cfg.model_name, llm_cfg.backend)
    llm = make_client(llm_cfg)

    log.info("Загружаю encoders...")
    pool = EncoderPool.from_config(cfg["encoders"])

    llm_judge = None
    if not args.skip_judge:
        from src.filters import LLMJudge
        llm_judge = LLMJudge(
            llm,
            mode="fast",
            temperature=cfg["llm"]["judge_temperature"],
            max_new_tokens_fast=cfg["llm"]["judge_max_new_tokens"],
        )
        log.info("LLM-judge: enabled (fast, stopping criterion)")
    else:
        log.info("LLM-judge: skipped")

    try:
        from tqdm import tqdm
        pbar = tqdm(examples, desc="GP iter", unit="ex")
    except ImportError:
        pbar = examples

    all_records: list[dict] = []
    for ex in pbar:
        recs = process_example(llm, pool, llm_judge, ex,
                               n_chains=args.n_chains,
                               n_iterations=args.n_iterations)
        all_records.extend(recs)
        if args.pilot:
            n_pass = sum(1 for r in recs if is_passed(r))
            depths = []
            for cidx in range(args.n_chains):
                chain = [r for r in recs if r.get("chain_idx") == cidx]
                if chain:
                    depths.append(max(r["iteration"] for r in chain))
            log.info("[%s] total=%d passed=%d depths=%s",
                     ex["qid"], len(recs), n_pass, depths)

    save_jsonl(out_dir / "candidates_raw.jsonl", all_records)
    save_jsonl(out_dir / "candidates_passed.jsonl",
               [r for r in all_records if is_passed(r)])

    summary = compute_summary(all_records, examples)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(summary, all_records, examples, out_dir, mode_name)

    print("\n" + "=" * 70)
    print(f"GP ITERATIVE — {mode_name.upper()}")
    print("=" * 70)
    cov  = summary["coverage"]
    cand = summary["candidates"]
    d = summary["chain_depth"]
    print(f"Input pairs:           {summary['n_input_pairs']}")
    print(f"Coverage (≥1 passed):  {cov['n_query_with_passed_candidate']}/{summary['n_input_pairs']} ({cov['pct_with_passed']}%)")
    print(f"Query w/ ≥2 levels:    {cov['n_query_with_2plus_levels']}")
    print(f"Query w/ ≥3 levels:    {cov['n_query_with_3plus_levels']}")
    print(f"Candidates total:      {cand['n_total']}, passed: {cand['n_passed']}")
    print(f"Chain depth (mean):    {d['mean_max_depth']:.2f}")
    print()
    print("Drift по итерациям:")
    print(f"  {'iter':>4} {'n':>4} {'sim_mean':>9} {'sim_median':>10} {'len_mean':>9} {'yes-rate':>9}")
    for k in sorted(summary["sim_by_iteration"].keys()):
        s = summary["sim_by_iteration"][k]
        l = summary["length_by_iteration"].get(k, {})
        y = summary["judge_yes_by_iteration"].get(k)
        y_str = f"{y:.1%}" if y is not None else "—"
        print(f"  {k:>4} {s['n']:>4} {s['mean']:>9.3f} {s['median']:>10.3f} "
              f"{l.get('mean', 0):>9.2f} {y_str:>9}")
    print()
    print("Bucket counts (passed):")
    for bucket, n in summary["bucket_counts_passed"].items():
        print(f"  {bucket}: {n}")
    print()
    if summary["stop_reasons"]:
        print("Stop reasons:")
        for r, n in sorted(summary["stop_reasons"].items(), key=lambda x: -x[1]):
            print(f"  {r}: {n}")
    print()
    print(f"Files:")
    print(f"  {out_dir / 'candidates_raw.jsonl'}")
    print(f"  {out_dir / 'candidates_passed.jsonl'}")
    print(f"  {out_dir / 'summary.json'}")
    print(f"  {out_dir / 'report.md'}")

    if args.pilot:
        print("\n" + "─" * 70)
        print("PILOT CHECKLIST (iterative GP)")
        print("─" * 70)
        sim_iter = summary["sim_by_iteration"]
        len_iter = summary["length_by_iteration"]
        means = [sim_iter[k]["mean"] for k in sorted(sim_iter.keys())]
        monotone = (all(means[i] >= means[i + 1] - 0.005 for i in range(len(means) - 1))
                    if len(means) >= 2 else False)
        len_k1 = len_iter.get(1, {}).get("mean", 0)
        mean_depth = summary["chain_depth"]["mean_max_depth"]

        checks = [
            ("Coverage ≥90%",
             cov["pct_with_passed"] >= 90, f"{cov['pct_with_passed']}%"),
            ("Mean sim_to_dp убывает с k (есть drift)",
             monotone and len(means) >= 2, f"means: {means}"),
            ("Length mean на iter=1 ∈ [0.7, 1.4]",
             0.7 <= len_k1 <= 1.4, f"{len_k1:.2f}"),
            ("Mean chain depth ≥2 (judge не режет всё на k=1)",
             mean_depth >= 2.0, f"{mean_depth:.2f}"),
            ("Query w/ ≥2 sim-levels ≥40%",
             cov["n_query_with_2plus_levels"] / max(1, summary["n_input_pairs"]) >= 0.4,
             f"{cov['n_query_with_2plus_levels']}/{summary['n_input_pairs']}"),
        ]
        for name, ok, val in checks:
            mark = "✓" if ok else "✗"
            print(f"  {mark}  {name}  →  {val}")
        if all(c[1] for c in checks):
            print("\n✓ Pilot OK — запускай production: --n-examples 500 (без --pilot)")
        else:
            print("\n⚠ Есть провалы. Смотри report.md → таблица 'Drift по итерациям'.")
            print("  Возможные причины: промпт не держит длину; judge режет слишком рано;")
            print("  модель повторяет ту же версию (no_change_from_prev).")


if __name__ == "__main__":
    main()
