#!/usr/bin/env python3
"""
Phase 2: δ-распределение CN на RuBQ-dev50 без фильтров.

ЦЕЛЬ. Получить реальное распределение δ = sim(q, d⁻) / sim(q, d⁺) на полноценных
русских данных, чтобы откалибровать границы бакетов easy/medium/hard в default.yaml.
Текущие границы (0.30/0.55/0.75/0.92) взяты с потолка из литературы по английским
данным; для русского RuBQ это может быть смещено.

ЧТО ДЕЛАЕМ.
  1. Загружаем dev50 (50 пар q/d⁺).
  2. Для каждой пары: extract_facts → top-K по criticality → mutate (один CF на факт).
  3. Считаем δ через ансамбль (e5-large + bge-m3).
  4. БЕЗ judges. Все CF идут в выгрузку с пометками match_status фактов.
  5. Аналитика: гистограмма δ, квантили, разбивка по fact_type, рекомендация бакетов.

ВРЕМЯ. На RTX 4060 + Q4 + Ollama:
  - extract_facts (50 вызовов): ~3-5 мин
  - mutate (50 × ~3 факта = ~150 вызовов): ~10-15 мин
  - encoders (загрузка ~5 мин кеша после первого раза + ~1 мин расчёта): ~6 мин
  - Итого: 20-30 минут на полный прогон.

ЗАПУСК.
    cd W:\\Jupyter\\NIR_2
    $env:HF_HOME = "W:\\huggingface_cache"
    python scripts\\03_delta_distribution.py

    # ограничить для отладки:
    python scripts\\03_delta_distribution.py --limit 10
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median, quantiles, stdev

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.llm_client import LLMConfig, make_client, extract_json  # noqa: E402
from src.prompts import (  # noqa: E402
    build_fact_extraction_messages,
    build_counterfactual_messages,
)
from src.text_utils import fuzzy_in, morph_available  # noqa: E402
from src.encoders import EncoderPool  # noqa: E402

log = logging.getLogger("delta_dist")


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


def _classify_fact(fact_text: str, d_plus: str) -> str:
    """Тот же классификатор что в фиксированном sanity-скрипте."""
    if fact_text in d_plus:
        return "verbatim"
    if fuzzy_in(fact_text, d_plus):
        return "morphological_variant"
    return "hallucinated"


# ──────────────────────────────────────────────────────────────────────
# Этап 1: extract facts с фильтром
# ──────────────────────────────────────────────────────────────────────

def run_fact_extraction(
    llm, examples: list[dict], min_criticality: int = 3, max_facts: int = 5
) -> list[dict]:
    """Возвращает по записи на каждый пример: query, d_plus, valid_facts."""
    log.info("=== Этап 1: Fact extraction (%d пар) ===", len(examples))

    try:
        from tqdm import tqdm
        pbar = tqdm(total=len(examples), desc="Extract", unit="ex")
    except ImportError:
        pbar = None

    results = []
    for ex in examples:
        msgs = build_fact_extraction_messages(ex["query"], ex["d_plus"])
        raw = llm.generate([msgs])[0]
        parsed = extract_json(raw)
        valid_json = isinstance(parsed, list)

        verbatim, morph, hallu = [], [], []
        if valid_json:
            for item in parsed:
                if not isinstance(item, dict) or "text" not in item:
                    continue
                fact = {
                    "text": str(item.get("text", "")).strip(),
                    "type": str(item.get("type", "")).strip(),
                    "criticality": int(item.get("criticality", 0)),
                }
                if fact["criticality"] < min_criticality:
                    continue
                status = _classify_fact(fact["text"], ex["d_plus"])
                fact["match_status"] = status
                if status == "verbatim":
                    verbatim.append(fact)
                elif status == "morphological_variant":
                    morph.append(fact)
                else:
                    hallu.append(fact)

        # Для CN берём только verbatim (по той же причине что в Phase 0.5):
        # LLM должна физически найти и подменить подстроку.
        verbatim.sort(key=lambda f: -f["criticality"])
        top = verbatim[:max_facts]

        results.append({
            "qid": ex["qid"],
            "query": ex["query"],
            "d_plus": ex["d_plus"],
            "valid_json": valid_json,
            "facts_top": top,
            "n_verbatim": len(verbatim),
            "n_morph": len(morph),
            "n_hallu": len(hallu),
            "tags": ex.get("meta", {}).get("tags", []),
        })
        if pbar:
            pbar.set_postfix(verb=len(verbatim), morph=len(morph))
            pbar.update(1)

    if pbar:
        pbar.close()
    return results


# ──────────────────────────────────────────────────────────────────────
# Этап 2: мутация (по одному CF на каждый top-факт)
# ──────────────────────────────────────────────────────────────────────

def run_mutations(llm, fact_results: list[dict]) -> list[dict]:
    log.info("=== Этап 2: Counterfactual mutation ===")

    pairs, batch_msgs = [], []
    for fr in fact_results:
        for fact in fr["facts_top"]:
            pairs.append((fr, fact))
            batch_msgs.append(build_counterfactual_messages(
                query=fr["query"], d_plus=fr["d_plus"],
                fact_text=fact["text"], fact_type=fact["type"],
            ))

    log.info("  всего мутаций: %d", len(batch_msgs))
    if not batch_msgs:
        return []

    try:
        from tqdm import tqdm
        iter_pairs = tqdm(list(zip(pairs, batch_msgs)), total=len(pairs),
                          desc="Mutate", unit="fact")
    except ImportError:
        iter_pairs = list(zip(pairs, batch_msgs))

    cfs = []
    for (fr, fact), msgs in iter_pairs:
        raw = llm.generate([msgs])[0]
        d_minus = raw.strip()
        if d_minus.startswith('"') and d_minus.endswith('"') and len(d_minus) > 2:
            d_minus = d_minus[1:-1].strip()

        is_self = d_minus == fr["d_plus"].strip()
        length_ratio = len(d_minus) / len(fr["d_plus"]) if fr["d_plus"] else 0.0
        fact_still_in = fact["text"] in d_minus

        cfs.append({
            "qid": fr["qid"],
            "query": fr["query"],
            "d_plus": fr["d_plus"],
            "d_minus": d_minus,
            "fact_text": fact["text"],
            "fact_type": fact["type"],
            "fact_criticality": fact["criticality"],
            "is_self_copy": is_self,
            "length_ratio": length_ratio,
            "fact_still_present": fact_still_in,
            "tags": fr["tags"],
        })
    return cfs


# ──────────────────────────────────────────────────────────────────────
# Этап 3: δ через ансамбль
# ──────────────────────────────────────────────────────────────────────

def compute_deltas(pool: EncoderPool, cfs: list[dict]) -> list[dict]:
    log.info("=== Этап 3: δ через ансамбль (%d энкодеров) ===", len(pool.encoders))

    # Фильтруем self-copies — для них δ бессмыслен (= 1.0 by definition)
    valid_cfs = [c for c in cfs if not c["is_self_copy"]]
    log.info("  CF к расчёту: %d (отброшено self_copy: %d)",
             len(valid_cfs), len(cfs) - len(valid_cfs))

    try:
        from tqdm import tqdm
        iter_cfs = tqdm(valid_cfs, desc="δ", unit="cf")
    except ImportError:
        iter_cfs = valid_cfs

    out = []
    for cf in iter_cfs:
        sims_agg, sims_per_enc = pool.similarity_batch(
            cf["query"], [cf["d_plus"], cf["d_minus"]], role="qd"
        )
        sim_pos, sim_neg = float(sims_agg[0]), float(sims_agg[1])

        per_enc_delta: dict[str, float] = {}
        for enc_name, sims in sims_per_enc.items():
            sp, sn = float(sims[0]), float(sims[1])
            per_enc_delta[enc_name] = sn / sp if abs(sp) > 1e-6 else 0.0
        delta_agg = mean(per_enc_delta.values())

        rec = dict(cf)
        rec["sim_q_dplus"] = sim_pos
        rec["sim_q_dminus"] = sim_neg
        rec["delta"] = delta_agg
        rec["delta_per_enc"] = per_enc_delta
        out.append(rec)
    return out


# ──────────────────────────────────────────────────────────────────────
# Аналитика и рекомендации
# ──────────────────────────────────────────────────────────────────────

def quantile_stats(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0}
    qs = quantiles(xs, n=10) if len(xs) >= 10 else [None] * 9
    return {
        "n": len(xs),
        "mean": round(mean(xs), 4),
        "median": round(median(xs), 4),
        "stdev": round(stdev(xs), 4) if len(xs) > 1 else 0.0,
        "min": round(min(xs), 4),
        "max": round(max(xs), 4),
        "p10": round(qs[0], 4) if qs[0] is not None else None,
        "p25": round(qs[1], 4) if qs[1] is not None else None,
        "p50": round(qs[4], 4) if qs[4] is not None else None,
        "p75": round(qs[6], 4) if qs[6] is not None else None,
        "p90": round(qs[8], 4) if qs[8] is not None else None,
    }


def analyze(deltas_records: list[dict], fact_results: list[dict]) -> dict:
    """Сводка по δ + по fact_types + рекомендация бакетов."""
    deltas = [r["delta"] for r in deltas_records]

    # По типу мутации (грубая категоризация)
    def fact_class(t: str) -> str:
        t = t.lower()
        if any(k in t for k in ("дат", "год", "период", "век")):  return "date"
        if any(k in t for k in ("число", "номер", "колич", "процент")):  return "number"
        if any(k in t for k in ("имя", "автор", "основат", "руководит")):  return "name"
        if any(k in t for k in ("страна", "город", "столиц", "местоположен", "географ")):  return "geo"
        if any(k in t for k in ("организ", "компан", "группа")):  return "organization"
        return "other"

    by_type: dict[str, list[float]] = defaultdict(list)
    for r in deltas_records:
        by_type[fact_class(r["fact_type"])].append(r["delta"])

    # Pipeline coverage
    n_examples = len(fact_results)
    n_zero_facts = sum(1 for fr in fact_results if not fr["facts_top"])
    facts_per_doc = [len(fr["facts_top"]) for fr in fact_results]

    # Self-copy rate (важно для качества генератора)
    # — считаем по всем cfs до фильтрации, поэтому возьмём из meta
    # (не приходит сюда; делаем грубую оценку: n_attempted − n_with_delta)
    # Это посчитаем в main и передадим отдельно — здесь оставим placeholder.

    summary = {
        "n_examples_processed": n_examples,
        "n_examples_with_zero_facts": n_zero_facts,
        "facts_per_doc": {
            "mean": round(mean(facts_per_doc), 2) if facts_per_doc else 0,
            "median": median(facts_per_doc) if facts_per_doc else 0,
            "min": min(facts_per_doc) if facts_per_doc else 0,
            "max": max(facts_per_doc) if facts_per_doc else 0,
        },
        "delta_overall": quantile_stats(deltas),
        "delta_by_fact_class": {k: quantile_stats(v) for k, v in by_type.items()},
    }

    # Рекомендация границ бакетов из квантилей:
    #   easy:   p10 .. p33 (~33% самых дальних)
    #   medium: p33 .. p67
    #   hard:   p67 .. p90 (~33% самых близких, но без экстремумов)
    #   discarded: δ > p90 (слишком близко к d⁺, ненадёжный негатив)
    if len(deltas) >= 10:
        # Используем встроенные квантили; здесь нам нужны 33% и 67%
        sorted_d = sorted(deltas)
        p33 = sorted_d[int(0.33 * len(sorted_d))]
        p67 = sorted_d[int(0.67 * len(sorted_d))]
        p10 = sorted_d[int(0.10 * len(sorted_d))]
        p90 = sorted_d[int(0.90 * len(sorted_d))]
        summary["bucket_recommendations"] = {
            "easy":   {"min": round(p10, 3), "max": round(p33, 3)},
            "medium": {"min": round(p33, 3), "max": round(p67, 3)},
            "hard":   {"min": round(p67, 3), "max": round(p90, 3)},
            "discarded_above": round(p90, 3),
            "rationale": "На основе квантилей наблюдаемого δ: easy=p10-p33, medium=p33-p67, hard=p67-p90. CF с δ>p90 отбрасываем как ненадёжные.",
        }
    return summary


# ──────────────────────────────────────────────────────────────────────
# Визуализация
# ──────────────────────────────────────────────────────────────────────

def make_plots(records: list[dict], output_dir: Path) -> Path | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib не установлен — пропускаю plots")
        return None

    deltas = [r["delta"] for r in records]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # (1) Общее распределение δ
    ax1 = axes[0]
    ax1.hist(deltas, bins=25, color="#264653", alpha=0.85, edgecolor="white")
    sorted_d = sorted(deltas)
    if len(deltas) >= 10:
        p33 = sorted_d[int(0.33 * len(sorted_d))]
        p67 = sorted_d[int(0.67 * len(sorted_d))]
        p90 = sorted_d[int(0.90 * len(sorted_d))]
        ax1.axvline(p33, color="#2a9d8f", linestyle="--", linewidth=1.5, label=f"p33={p33:.2f}")
        ax1.axvline(p67, color="#e9c46a", linestyle="--", linewidth=1.5, label=f"p67={p67:.2f}")
        ax1.axvline(p90, color="#e76f51", linestyle="--", linewidth=1.5, label=f"p90={p90:.2f} (discard above)")
    ax1.set_xlabel("δ = sim(q, d⁻) / sim(q, d⁺)")
    ax1.set_ylabel("количество CF")
    ax1.set_title(f"Распределение δ на dev50\nn={len(deltas)} CF")
    ax1.legend(loc="upper left", fontsize=9)

    # (2) δ по классам мутации
    ax2 = axes[1]
    by_type: dict[str, list[float]] = defaultdict(list)
    for r in records:
        t = r["fact_type"].lower()
        if any(k in t for k in ("дат", "год", "период")):  cls = "date"
        elif any(k in t for k in ("число", "номер", "колич")):  cls = "number"
        elif any(k in t for k in ("имя", "автор", "основат")):  cls = "name"
        elif any(k in t for k in ("страна", "город", "столиц", "географ")):  cls = "geo"
        elif any(k in t for k in ("организ", "компан")):  cls = "organization"
        else:  cls = "other"
        by_type[cls].append(r["delta"])

    classes = sorted(by_type.keys(), key=lambda k: -len(by_type[k]))[:6]
    data = [by_type[c] for c in classes]
    labels = [f"{c}\n(n={len(by_type[c])})" for c in classes]
    bp = ax2.boxplot(data, labels=labels, patch_artist=True, widths=0.6)
    for patch, color in zip(bp["boxes"], ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51", "#8d99ae"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.set_ylabel("δ")
    ax2.set_title("δ по классам мутации")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = output_dir / "delta_distribution.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs/default.yaml"))
    ap.add_argument("--dev-set", default=str(PROJECT_ROOT / "data/rubq/rubq_dev50.jsonl"))
    ap.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs/phase2_delta"))
    ap.add_argument("--limit", type=int, default=None,
                    help="ограничить число примеров (для отладки)")
    ap.add_argument("--max-facts", type=int, default=3,
                    help="сколько top-фактов мутировать на пример (default: 3)")
    ap.add_argument("--min-criticality", type=int, default=3,
                    help="минимальная criticality факта для мутации")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    examples = load_jsonl(Path(args.dev_set))
    if args.limit:
        examples = examples[: args.limit]
    log.info("Загружено: %d пар q/d⁺", len(examples))
    log.info("pymorphy3: %s", "доступен" if morph_available() else "НЕТ")

    # LLM
    llm_cfg = LLMConfig.from_dict(cfg["llm"])
    log.info("LLM: %s (backend=%s)", llm_cfg.model_name, llm_cfg.backend)
    llm = make_client(llm_cfg)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Этап 1: extract
    fact_results = run_fact_extraction(
        llm, examples,
        min_criticality=args.min_criticality,
        max_facts=args.max_facts,
    )
    save_jsonl(out_dir / "facts.jsonl", fact_results)

    n_zero = sum(1 for fr in fact_results if not fr["facts_top"])
    log.info("  пар с 0 verbatim-фактов: %d/%d", n_zero, len(fact_results))

    # Этап 2: mutate
    cfs = run_mutations(llm, fact_results)
    save_jsonl(out_dir / "counterfactuals_raw.jsonl", cfs)

    n_self = sum(1 for c in cfs if c["is_self_copy"])
    log.info("  CF total: %d, self_copy: %d, fact_still_in: %d",
             len(cfs), n_self, sum(1 for c in cfs if c["fact_still_present"]))

    # Этап 3: encoders + δ
    log.info("Загружаю ансамбль энкодеров…")
    pool = EncoderPool.from_config(cfg["encoders"])
    deltas_records = compute_deltas(pool, cfs)
    save_jsonl(out_dir / "deltas.jsonl", deltas_records)

    # Аналитика + бакеты
    summary = analyze(deltas_records, fact_results)
    summary["counterfactual_generation"] = {
        "n_attempted": len(cfs),
        "n_self_copy": n_self,
        "n_with_delta": len(deltas_records),
        "fact_still_present_rate": round(
            sum(1 for c in cfs if c["fact_still_present"]) / max(1, len(cfs)), 4
        ),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    plot_path = make_plots(deltas_records, out_dir)

    # ──────────────────────────────────────────────
    # Печать сводки
    # ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 2 — δ-DISTRIBUTION SUMMARY")
    print("=" * 70)
    print(f"Examples processed:     {summary['n_examples_processed']}")
    print(f"Examples with 0 facts:  {summary['n_examples_with_zero_facts']}  "
          f"({summary['n_examples_with_zero_facts']/max(1,summary['n_examples_processed']):.0%})")
    fpd = summary["facts_per_doc"]
    print(f"Facts per doc:          mean={fpd['mean']:.2f}  median={fpd['median']}  range=[{fpd['min']}, {fpd['max']}]")
    print()
    cg = summary["counterfactual_generation"]
    print(f"CF attempted:           {cg['n_attempted']}")
    print(f"CF self-copy:           {cg['n_self_copy']}")
    print(f"CF with δ:              {cg['n_with_delta']}")
    print(f"Fact still in d⁻ rate:  {cg['fact_still_present_rate']:.1%}")
    print()
    do = summary["delta_overall"]
    print(f"δ overall (n={do['n']}):")
    print(f"  mean={do['mean']:.3f}  median={do['median']:.3f}  stdev={do['stdev']:.3f}  range=[{do['min']:.3f}, {do['max']:.3f}]")
    print(f"  quantiles: p10={do['p10']:.3f}  p25={do['p25']:.3f}  p50={do['p50']:.3f}  p75={do['p75']:.3f}  p90={do['p90']:.3f}")
    print()
    print("δ by fact class:")
    for cls, s in sorted(summary["delta_by_fact_class"].items(), key=lambda x: -x[1]["n"]):
        if s["n"]:
            print(f"  {cls:13s}: n={s['n']:>3}  mean={s['mean']:.3f}  median={s['median']:.3f}")
    print()
    if "bucket_recommendations" in summary:
        br = summary["bucket_recommendations"]
        print("RECOMMENDED BUCKET BOUNDARIES (для default.yaml):")
        for name in ("easy", "medium", "hard"):
            b = br[name]
            print(f"  {name:6s}: [{b['min']:.3f}, {b['max']:.3f}]")
        print(f"  discard CF with δ > {br['discarded_above']:.3f}")
        print()
        # сравнение с дефолтом
        default_buckets = cfg["negatives"]["delta_buckets"]
        print("Current default.yaml buckets (для сравнения):")
        for b in default_buckets:
            print(f"  {b['name']:6s}: [{b['min']:.3f}, {b['max']:.3f}]")
    print()
    print(f"Files:")
    print(f"  {out_dir / 'facts.jsonl'}")
    print(f"  {out_dir / 'counterfactuals_raw.jsonl'}")
    print(f"  {out_dir / 'deltas.jsonl'}")
    print(f"  {out_dir / 'summary.json'}")
    if plot_path:
        print(f"  {plot_path}")


if __name__ == "__main__":
    main()
