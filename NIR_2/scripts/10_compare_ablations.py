#!/usr/bin/env python3
"""
Phase PoC: финальное сравнение 5 ablation conditions.

ЧТО ЭТО.
  Агрегирует результаты от 09_eval_retriever (по одной модели за раз)
  в единую сравнительную таблицу. Считает deltas от baseline (A),
  проверяет PoC SUCCESS condition (C - A ≥ 1.5 pt NDCG@10), делает
  per-query flip analysis.

ВХОД.
  outputs/eval/<tag>/summary.json     — от 09 для каждой модели.
  outputs/eval/<tag>/per_query.jsonl  — для per-query flip analysis.

ВЫХОД.
  outputs/eval/comparison.json
  outputs/eval/comparison.md
  outputs/eval/comparison_chart.png  (опционально, если matplotlib доступен)

РЕШЕНИЕ ПО PoC (из PROJECT_STATE).
  PoC SUCCESS = условие C побеждает baseline A на ≥1.5 pt NDCG@10.
  Дополнительные индикаторы:
    - C ≥ B (CN лучше BM25-стандарта)
    - D > A (GP augmentation помогает)
    - E максимум (композиция самая сильная)

CAVEAT для defence.
  Эти числа — с одного seed. Effect size 1.5-2 pt может быть внутри
  single-seed noise. Multi-seed runs нужны для надёжного CI. Это явно
  пометится в финальном report.md.

ЗАПУСК.
  python scripts/10_compare_ablations.py
  python scripts/10_compare_ablations.py --tags A_seed42 B_seed42 C_seed42 D_seed42 E_seed42
  python scripts/10_compare_ablations.py --include-baseline BASE
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

log = logging.getLogger("compare")

# Метрики которые мы хотим сравнить (по убыванию приоритета для PoC)
PRIMARY_METRIC = "ndcg@10"
COMPARE_METRICS = ["ndcg@10", "mrr@10", "recall@10", "recall@5", "recall@1"]


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


def discover_eval_tags(eval_base: Path, seed: int) -> list[str]:
    """Найти все эвал-теги в outputs/eval/. По умолчанию ищем условные _seed42."""
    if not eval_base.exists():
        return []
    tags = []
    for sub in sorted(eval_base.iterdir()):
        if sub.is_dir() and (sub / "summary.json").exists():
            tags.append(sub.name)
    return tags


# ──────────────────────────────────────────────────────────────────────
# Aggregation
# ──────────────────────────────────────────────────────────────────────

def load_results(eval_base: Path, tags: list[str]) -> dict[str, dict]:
    """tag → {summary, per_query}"""
    out = {}
    for tag in tags:
        sd = eval_base / tag
        sumf = sd / "summary.json"
        pqf = sd / "per_query.jsonl"
        if not sumf.exists():
            log.warning("[%s] нет summary.json — пропускаю", tag)
            continue
        summary = json.loads(sumf.read_text())
        per_query = load_jsonl(pqf) if pqf.exists() else []
        out[tag] = {"summary": summary, "per_query": per_query}
    return out


def compute_deltas(
    results: dict[str, dict],
    baseline_tag: str,
) -> dict[str, dict]:
    """
    Для каждого tag (кроме baseline) считает (metric - metric_baseline)
    × 100 в pt.
    """
    if baseline_tag not in results:
        return {}
    base_m = results[baseline_tag]["summary"]["metrics"]
    deltas = {}
    for tag, r in results.items():
        m = r["summary"]["metrics"]
        d = {}
        for key in COMPARE_METRICS:
            if key in m and key in base_m:
                d[key] = round(100 * (m[key] - base_m[key]), 2)  # в pt (×100)
        deltas[tag] = d
    return deltas


def flip_analysis(
    results: dict[str, dict],
    baseline_tag: str,
    target_tag: str,
) -> dict:
    """
    Per-query analysis: какие queries flipped baseline → target.
      - hit→hit (rank change?)
      - miss→hit: что target нашёл, а baseline пропустил
      - hit→miss: что baseline нашёл, а target пропустил (регрессия)
      - miss→miss
    """
    if baseline_tag not in results or target_tag not in results:
        return {}
    base_pq = {q["qid"]: q for q in results[baseline_tag]["per_query"]}
    tgt_pq = {q["qid"]: q for q in results[target_tag]["per_query"]}
    common_qids = set(base_pq) & set(tgt_pq)

    miss_to_hit = []
    hit_to_miss = []
    hit_to_hit_better = 0
    hit_to_hit_worse = 0
    hit_to_hit_same = 0
    miss_to_miss = 0

    for qid in common_qids:
        b, t = base_pq[qid], tgt_pq[qid]
        b_hit = b["gold_in_top_k"]
        t_hit = t["gold_in_top_k"]
        if not b_hit and not t_hit:
            miss_to_miss += 1
        elif not b_hit and t_hit:
            miss_to_hit.append({
                "qid": qid, "query": b["query"][:120],
                "tgt_rank": t["gold_rank"],
            })
        elif b_hit and not t_hit:
            hit_to_miss.append({
                "qid": qid, "query": b["query"][:120],
                "base_rank": b["gold_rank"],
            })
        else:
            if t["gold_rank"] < b["gold_rank"]:
                hit_to_hit_better += 1
            elif t["gold_rank"] > b["gold_rank"]:
                hit_to_hit_worse += 1
            else:
                hit_to_hit_same += 1

    return {
        "baseline": baseline_tag,
        "target": target_tag,
        "n_common_queries": len(common_qids),
        "miss_to_hit": len(miss_to_hit),
        "hit_to_miss": len(hit_to_miss),
        "hit_to_hit_better": hit_to_hit_better,
        "hit_to_hit_worse": hit_to_hit_worse,
        "hit_to_hit_same": hit_to_hit_same,
        "miss_to_miss": miss_to_miss,
        "net_improvement": len(miss_to_hit) - len(hit_to_miss),
        "examples_miss_to_hit": miss_to_hit[:5],
        "examples_hit_to_miss": hit_to_miss[:5],
    }


# ──────────────────────────────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────────────────────────────

def write_report(
    results: dict[str, dict],
    deltas: dict[str, dict],
    flips: dict[str, dict],
    baseline_tag: str,
    poc_threshold: float,
    output_dir: Path,
) -> None:
    lines = []
    lines.append("# Ablation Comparison — PoC Final Results\n")
    lines.append(f"- Baseline: **{baseline_tag}**")
    lines.append(f"- Primary metric: **{PRIMARY_METRIC}**")
    lines.append(f"- PoC SUCCESS threshold (C - A on {PRIMARY_METRIC}): "
                 f"**≥ {poc_threshold} pt**")
    lines.append("")

    # ── Main table ──
    lines.append("## Main metrics\n")
    header = "| condition |" + "|".join(f" {m} " for m in COMPARE_METRICS) + "|"
    sep    = "|---|" + "|".join("---:" for _ in COMPARE_METRICS) + "|"
    lines.append(header)
    lines.append(sep)
    for tag in sorted(results.keys()):
        m = results[tag]["summary"]["metrics"]
        row = f"| **{tag}** |"
        for key in COMPARE_METRICS:
            v = m.get(key)
            row += f" {v:.4f} |" if isinstance(v, (int, float)) else " — |"
        lines.append(row)
    lines.append("")

    # ── Deltas от baseline (в pt) ──
    lines.append(f"## Δ from baseline ({baseline_tag}) — в pt (×100)\n")
    header = "| condition |" + "|".join(f" Δ{m} " for m in COMPARE_METRICS) + "|"
    sep    = "|---|" + "|".join("---:" for _ in COMPARE_METRICS) + "|"
    lines.append(header)
    lines.append(sep)
    for tag in sorted(deltas.keys()):
        d = deltas[tag]
        row = f"| **{tag}** |"
        for key in COMPARE_METRICS:
            v = d.get(key)
            if v is None:
                row += " — |"
            else:
                sign = "+" if v > 0 else ""
                row += f" {sign}{v:.2f} |"
        lines.append(row)
    lines.append("")

    # ── PoC decision ──
    lines.append("## PoC Decision\n")
    poc_target_tags = [t for t in deltas if t.startswith("C")]
    poc_target = poc_target_tags[0] if poc_target_tags else None
    if poc_target and PRIMARY_METRIC in deltas[poc_target]:
        delta_c = deltas[poc_target][PRIMARY_METRIC]
        verdict = "**✅ SUCCESS**" if delta_c >= poc_threshold else "**❌ NOT MET**"
        lines.append(f"- C - A on {PRIMARY_METRIC}: **{delta_c:+.2f} pt** "
                     f"(threshold {poc_threshold} pt) — {verdict}")
    else:
        lines.append("- Condition C не найден среди eval results.")
    lines.append("")

    # Дополнительные индикаторы
    lines.append("### Дополнительные индикаторы\n")
    lines.append("| Comparison | Δ on ndcg@10 | Interpretation |")
    lines.append("|---|---:|---|")
    pairs = [
        ("C vs B", "C", "B", "CN vs BM25 baseline (главный вопрос)"),
        ("D vs A", "D", "A", "GP augmentation полезен?"),
        ("E vs C", "E", "C", "GP добавляет к CN?"),
        ("E vs A", "E", "A", "Композиция против чистого baseline"),
    ]
    for label, lhs, rhs, interp in pairs:
        lhs_tag = next((t for t in deltas if t.startswith(lhs)), None)
        rhs_tag = next((t for t in deltas if t.startswith(rhs)), None)
        if lhs_tag and rhs_tag:
            lhs_m = results[lhs_tag]["summary"]["metrics"].get(PRIMARY_METRIC)
            rhs_m = results[rhs_tag]["summary"]["metrics"].get(PRIMARY_METRIC)
            if lhs_m is not None and rhs_m is not None:
                delta = round(100 * (lhs_m - rhs_m), 2)
                lines.append(f"| {label} | {delta:+.2f} | {interp} |")
    lines.append("")

    # ── Flip analysis ──
    if flips:
        lines.append("## Per-query flip analysis\n")
        for key, flip in flips.items():
            lines.append(f"### {key}\n")
            lines.append(f"- common queries: {flip['n_common_queries']}")
            lines.append(f"- **miss → hit**: {flip['miss_to_hit']} "
                         f"(target нашёл то, что baseline пропустил)")
            lines.append(f"- **hit → miss**: {flip['hit_to_miss']} "
                         f"(target пропустил то, что baseline нашёл — регрессия)")
            lines.append(f"- hit→hit (better rank): {flip['hit_to_hit_better']}")
            lines.append(f"- hit→hit (worse rank): {flip['hit_to_hit_worse']}")
            lines.append(f"- hit→hit (same):       {flip['hit_to_hit_same']}")
            lines.append(f"- miss→miss:            {flip['miss_to_miss']}")
            lines.append(f"- **net improvement** (miss→hit minus hit→miss): "
                         f"**{flip['net_improvement']:+d}**")

            if flip["examples_miss_to_hit"]:
                lines.append(f"\n**Примеры miss→hit:**")
                for ex in flip["examples_miss_to_hit"]:
                    lines.append(f"- {ex['qid']} (rank {ex['tgt_rank']}): {ex['query']}")
            if flip["examples_hit_to_miss"]:
                lines.append(f"\n**Примеры hit→miss (регрессии):**")
                for ex in flip["examples_hit_to_miss"]:
                    lines.append(f"- {ex['qid']} (baseline rank {ex['base_rank']}): {ex['query']}")
            lines.append("")

    # ── Caveats ──
    lines.append("---\n## Caveats (для НИР-defense)\n")
    lines.append("### Single seed")
    lines.append(f"Все цифры получены с **одним seed**. Effect size в "
                 f"районе порога ({poc_threshold} pt NDCG@10) **может быть "
                 f"внутри single-seed noise**. Для надёжного confidence "
                 f"interval нужно 3+ seeds × 5 conditions = 15 прогонов. "
                 f"Это отложено до момента когда compute time позволит.\n")
    lines.append("### Curriculum learning")
    lines.append("Текущий PoC — **data ablation**, не curriculum. Все "
                 "conditions подают данные shuffled, без ordering по "
                 "difficulty. Curriculum фаза (F-condition: GP positives "
                 "ordered by sim_to_d_plus или sequential epochs BM25→CN) "
                 "идёт ПОСЛЕ validation отдельных компонент. Если "
                 "результаты PoC валидны → curriculum как продолжение работы.")
    lines.append("")
    lines.append("### Test set ограничения")
    n_test = (next(iter(results.values()))["summary"]["data"]["n_test_queries"]
              if results else 0)
    lines.append(f"- n_test = {n_test} queries. На малом тесте малая дельта "
                 f"может быть статистически незначимой.")
    lines.append(f"- Corpus 2096 docs — небольшой по retrieval-стандартам. "
                 f"Effect size может уменьшиться на большем корпусе.")
    lines.append(f"- RuBQ как датасет имеет ~24% пар с слабой связью q↔d⁺ "
                 f"(Phase 2 finding). Это вносит шум во все условия одинаково, "
                 f"но снижает overall metrics.")
    lines.append("")

    (output_dir / "comparison.md").write_text("\n".join(lines), encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────
# Optional chart
# ──────────────────────────────────────────────────────────────────────

def try_plot_chart(
    results: dict[str, dict],
    baseline_tag: str,
    output_path: Path,
) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.info("matplotlib недоступен — пропускаю chart")
        return False

    tags = sorted(results.keys())
    values = [results[t]["summary"]["metrics"].get(PRIMARY_METRIC, 0) for t in tags]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(tags, values, color=["#888"] + ["#3a7bd5"] * (len(tags) - 1))
    # Подсветить baseline другим цветом
    for i, t in enumerate(tags):
        if t == baseline_tag:
            bars[i].set_color("#888")
        elif t.startswith("C"):
            bars[i].set_color("#d04848")  # CN — главная гипотеза, выделим
    ax.axhline(values[tags.index(baseline_tag)] if baseline_tag in tags else 0,
               linestyle="--", color="#888", alpha=0.5, label="baseline")
    ax.set_ylabel(PRIMARY_METRIC)
    ax.set_title("Ablation comparison")
    ax.set_ylim(bottom=0)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()
    log.info("Chart saved: %s", output_path)
    return True


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-base",
                    default=str(PROJECT_ROOT / "outputs/eval"))
    ap.add_argument("--tags", nargs="+", default=None,
                    help="конкретные теги для сравнения; по умолчанию авто-discovery")
    ap.add_argument("--baseline-tag", default=None,
                    help="имя tag для baseline (default: первый с префиксом 'A_')")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--poc-threshold", type=float, default=1.5,
                    help="Δ ndcg@10 (pt) для PoC SUCCESS (default 1.5)")
    ap.add_argument("--flip-pairs", nargs="*", default=None,
                    help="пары baseline:target для flip analysis "
                         "(например 'A_seed42:C_seed42'). "
                         "По умолчанию: все vs baseline.")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )

    eval_base = Path(args.eval_base)
    if args.tags:
        tags = args.tags
    else:
        tags = discover_eval_tags(eval_base, args.seed)
        log.info("Auto-discovered tags: %s", tags)

    if not tags:
        log.error("Не найдены eval results в %s. Сначала запусти 09_eval_retriever.py", eval_base)
        sys.exit(1)

    results = load_results(eval_base, tags)
    if not results:
        log.error("Не удалось загрузить eval results")
        sys.exit(1)

    # Определяем baseline
    if args.baseline_tag:
        baseline_tag = args.baseline_tag
    else:
        # Default: tag начинающийся с "A_" (т.е. condition A с seed)
        a_tags = [t for t in results if t.startswith("A_") or t == "A"]
        if not a_tags:
            log.warning("Не найден A condition — использую первый tag как baseline")
            baseline_tag = sorted(results.keys())[0]
        else:
            baseline_tag = sorted(a_tags)[0]
    log.info("Baseline: %s", baseline_tag)

    # Compute deltas
    deltas = compute_deltas(results, baseline_tag)

    # Flip analysis (по умолчанию все vs baseline)
    flips = {}
    if args.flip_pairs:
        for pair in args.flip_pairs:
            lhs, rhs = pair.split(":")
            key = f"{lhs} → {rhs}"
            flips[key] = flip_analysis(results, lhs, rhs)
    else:
        for tag in results:
            if tag == baseline_tag:
                continue
            key = f"{baseline_tag} → {tag}"
            flips[key] = flip_analysis(results, baseline_tag, tag)

    # Save outputs
    output_dir = eval_base
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON dump
    comparison = {
        "baseline_tag": baseline_tag,
        "poc_threshold_pt": args.poc_threshold,
        "metrics": {tag: r["summary"]["metrics"] for tag, r in results.items()},
        "deltas_from_baseline_pt": deltas,
        "flips": flips,
    }
    (output_dir / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    write_report(results, deltas, flips, baseline_tag, args.poc_threshold, output_dir)
    chart_made = try_plot_chart(results, baseline_tag,
                                output_dir / "comparison_chart.png")

    # ── Stdout ──
    print("\n" + "=" * 80)
    print("ABLATION COMPARISON")
    print("=" * 80)
    print(f"Baseline: {baseline_tag}")
    print(f"PoC SUCCESS threshold (C - A on {PRIMARY_METRIC}): ≥ {args.poc_threshold} pt")
    print()
    print(f"{'tag':<22} " + " ".join(f"{m:>11}" for m in COMPARE_METRICS))
    for tag in sorted(results.keys()):
        m = results[tag]["summary"]["metrics"]
        row = f"{tag:<22} " + " ".join(
            f"{m.get(k, 0):>11.4f}" if isinstance(m.get(k), (int, float)) else f"{'—':>11}"
            for k in COMPARE_METRICS
        )
        print(row)
    print()
    print(f"Δ from {baseline_tag} (pt):")
    print(f"{'tag':<22} " + " ".join(f"{'Δ'+m:>11}" for m in COMPARE_METRICS))
    for tag in sorted(deltas.keys()):
        if tag == baseline_tag:
            continue
        d = deltas[tag]
        row = f"{tag:<22} " + " ".join(
            f"{d.get(k, 0):>+11.2f}" if isinstance(d.get(k), (int, float)) else f"{'—':>11}"
            for k in COMPARE_METRICS
        )
        print(row)
    print()

    # PoC verdict
    poc_target = next((t for t in deltas if t.startswith("C")), None)
    if poc_target and PRIMARY_METRIC in deltas[poc_target]:
        delta_c = deltas[poc_target][PRIMARY_METRIC]
        verdict = "✅ SUCCESS" if delta_c >= args.poc_threshold else "❌ NOT MET"
        print(f"PoC verdict (C - A on {PRIMARY_METRIC}): {delta_c:+.2f} pt → {verdict}")
    print()
    print("Files:")
    print(f"  {output_dir / 'comparison.json'}")
    print(f"  {output_dir / 'comparison.md'}")
    if chart_made:
        print(f"  {output_dir / 'comparison_chart.png'}")


if __name__ == "__main__":
    main()
