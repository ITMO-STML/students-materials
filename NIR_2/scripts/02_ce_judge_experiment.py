#!/usr/bin/env python3
"""
Phase 0.6: CE-эксперимент на контрфактах из sanity-набора.

ЛОГИКА. CoT LLM-judge показал 41.9% accuracy на counterfactuals (хуже Run 1 fast judge).
Все 18 ошибок — false positives: judge confabulates на ШАГЕ 3, оправдывая числовые
и датовые подмены, потому что у Qwen2.5-7B слабый factual recall на русском tail-знании.

Гипотеза: cross-encoder ансамбль (e5-large + bge-m3) НЕ имеет prior knowledge о
Куликовской битве или количестве спутников Юпитера — он смотрит чисто на семантическую
близость query↔doc. Числовая подмена смещает эмбеддинг, что должно дать сигнал.

Метрика: δ = sim(q, d⁻) / sim(q, d⁺) ∈ [0, 1.x]
  - δ ≈ 1.0 → d⁻ так же близок к запросу как d⁺ → hard CF (модели тяжело отличить)
  - δ ≈ 0.5 → d⁻ заметно слабее d⁺ → easy CF
  - δ < 0 → d⁻ противоречит запросу даже на уровне эмбеддинга (вряд ли случится)

Что проверяем:
  1. Распределение δ по 31 контрфакту (общее, и по типам мутации: дата / число / имя).
  2. Корреляция δ с judge-correctness: ловит ли высокий δ те случаи, где LLM ошибается?
  3. Согласованность двух энкодеров (e5 vs bge-m3 — насколько коррелированы δ?).
  4. Threshold: при каком δ можно сказать «слишком похож на d⁺, отбрасываем как
     ненадёжный негатив»?

Запуск:
    python scripts/02_ce_judge_experiment.py
    # или с другими входами:
    python scripts/02_ce_judge_experiment.py \
        --counterfactuals outputs/sanity/counterfactuals.jsonl \
        --judge outputs/sanity/judge.jsonl \
        --output-dir outputs/sanity/ce_experiment
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.encoders import EncoderPool  # noqa: E402

log = logging.getLogger("ce_experiment")


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
# Merge CF ↔ judge verdicts
# ──────────────────────────────────────────────────────────────────────

def match_judge_verdicts(
    cfs: list[dict], verdicts: list[dict]
) -> list[dict | None]:
    """
    Для каждого CF найти соответствующий verdict (kind='counterfactual').

    Матчим по (qid, doc==d_minus). Возвращаем list длины len(cfs), где None означает
    «verdict не нашёлся» (например, was self_copy и пропущен в run_judge).
    """
    cf_verdicts = [v for v in verdicts if v["kind"] == "counterfactual"]

    # Индекс по (qid, doc) — может быть несколько CF на один qid, у каждого свой d⁻
    index: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for v in cf_verdicts:
        index[(v["qid"], v["doc"])].append(v)

    out = []
    for cf in cfs:
        key = (cf["qid"], cf["d_minus"])
        matches = index.get(key, [])
        if matches:
            out.append(matches[0])
            # Если матчей несколько (один qid + один d⁻ судили дважды — не должно быть)
            # — используем первый, остальные просто не привязываются.
            matches.pop(0)
        else:
            out.append(None)
    n_unmatched = sum(1 for v in out if v is None)
    if n_unmatched:
        log.warning("Не сматчилось judge-вердиктов: %d из %d CF", n_unmatched, len(cfs))
    return out


# ──────────────────────────────────────────────────────────────────────
# Подсчёт δ через EncoderPool
# ──────────────────────────────────────────────────────────────────────

def compute_deltas(
    pool: EncoderPool, cfs: list[dict], verdicts: list[dict | None]
) -> list[dict]:
    """
    Для каждого CF посчитать sim(q, d⁺), sim(q, d⁻), δ (ансамблевые + per-encoder).
    Приклеить judge verdict.
    """
    log.info("Считаю δ для %d CF на ансамбле из %d энкодеров",
             len(cfs), len(pool.encoders))

    try:
        from tqdm import tqdm
        iter_cfs = tqdm(list(zip(cfs, verdicts)), total=len(cfs),
                        desc="CE δ", unit="cf")
    except ImportError:
        iter_cfs = list(zip(cfs, verdicts))

    records = []
    for cf, verdict in iter_cfs:
        q, dp, dm = cf["query"], cf["d_plus"], cf["d_minus"]

        # sim(q, d⁺) и sim(q, d⁻) per-encoder и ансамблем
        # Используем similarity_batch: один anchor=q, два candidate=[d⁺, d⁻].
        # Это эффективнее: один encode для q, один батч для двух d.
        sims_agg, sims_per_enc = pool.similarity_batch(q, [dp, dm], role="qd")
        sim_pos_agg, sim_neg_agg = float(sims_agg[0]), float(sims_agg[1])

        # Per-encoder δ
        per_enc_delta: dict[str, float] = {}
        per_enc_sim_pos: dict[str, float] = {}
        per_enc_sim_neg: dict[str, float] = {}
        for enc_name, sims in sims_per_enc.items():
            sp, sn = float(sims[0]), float(sims[1])
            per_enc_sim_pos[enc_name] = sp
            per_enc_sim_neg[enc_name] = sn
            per_enc_delta[enc_name] = sn / sp if abs(sp) > 1e-6 else 0.0

        # Ансамблевая δ: усредняем per-encoder δ (а не sim_pos/sim_neg отдельно)
        delta_agg = mean(per_enc_delta.values())

        rec = {
            "qid": cf["qid"],
            "query": q,
            "fact_text": cf["fact_text"],
            "fact_type": cf["fact_type"],
            "fact_criticality": cf["fact_criticality"],
            "d_plus_preview": dp[:80],
            "d_minus_preview": dm[:80],
            "sim_q_dplus_ensemble": sim_pos_agg,
            "sim_q_dminus_ensemble": sim_neg_agg,
            "delta_ensemble": delta_agg,
            "sim_q_dplus_per_enc": per_enc_sim_pos,
            "sim_q_dminus_per_enc": per_enc_sim_neg,
            "delta_per_enc": per_enc_delta,
        }
        if verdict is not None:
            rec["judge_verdict"] = verdict["verdict"]
            rec["judge_correct"] = verdict["correct"]
            rec["judge_reasoning"] = verdict.get("reasoning", "")
        else:
            rec["judge_verdict"] = None
            rec["judge_correct"] = None
            rec["judge_reasoning"] = ""
        records.append(rec)

    return records


# ──────────────────────────────────────────────────────────────────────
# Анализ
# ──────────────────────────────────────────────────────────────────────

def analyze(records: list[dict]) -> dict:
    """Сводка: общая статистика + split по judge-correct + по fact_type."""
    deltas = [r["delta_ensemble"] for r in records]
    deltas_correct = [r["delta_ensemble"] for r in records if r.get("judge_correct") is True]
    deltas_wrong   = [r["delta_ensemble"] for r in records if r.get("judge_correct") is False]

    # Per-encoder корреляция: насколько δ_e5 и δ_bge согласованы
    enc_names = list(records[0]["delta_per_enc"].keys()) if records else []
    per_enc_deltas: dict[str, list[float]] = {n: [] for n in enc_names}
    for r in records:
        for n in enc_names:
            per_enc_deltas[n].append(r["delta_per_enc"][n])

    # Pearson-like correlation между двумя энкодерами (если их два)
    enc_corr = None
    if len(enc_names) == 2:
        a, b = per_enc_deltas[enc_names[0]], per_enc_deltas[enc_names[1]]
        ma, mb = mean(a), mean(b)
        num = sum((ai - ma) * (bi - mb) for ai, bi in zip(a, b))
        sa = (sum((ai - ma) ** 2 for ai in a)) ** 0.5
        sb = (sum((bi - mb) ** 2 for bi in b)) ** 0.5
        enc_corr = num / (sa * sb) if (sa * sb) > 0 else None

    # По типам мутации
    by_type: dict[str, list[float]] = defaultdict(list)
    for r in records:
        # Унифицируем подкатегории fact_type в крупные классы для анализа
        t = r["fact_type"].lower()
        if any(k in t for k in ("дат", "год", "период")):
            cls = "date"
        elif any(k in t for k in ("число", "номер", "количеств")):
            cls = "number"
        elif any(k in t for k in ("имя", "автор", "основател")):
            cls = "name"
        elif any(k in t for k in ("столиц", "страна", "город", "месторасположен", "ocean", "оcean", "location")):
            cls = "geo"
        else:
            cls = "other"
        by_type[cls].append(r["delta_ensemble"])

    def stats(xs: list[float]) -> dict:
        if not xs:
            return {"n": 0}
        return {
            "n": len(xs),
            "mean": round(mean(xs), 4),
            "median": round(median(xs), 4),
            "stdev": round(stdev(xs), 4) if len(xs) > 1 else 0.0,
            "min": round(min(xs), 4),
            "max": round(max(xs), 4),
        }

    return {
        "overall": stats(deltas),
        "judge_correct_subgroup": stats(deltas_correct),
        "judge_wrong_subgroup": stats(deltas_wrong),
        "by_mutation_type": {k: stats(v) for k, v in by_type.items()},
        "encoder_correlation_pearson": round(enc_corr, 4) if enc_corr is not None else None,
        "encoder_names": enc_names,
    }


# ──────────────────────────────────────────────────────────────────────
# Визуализация
# ──────────────────────────────────────────────────────────────────────

def make_plots(records: list[dict], output_dir: Path) -> Path | None:
    """Гистограммы + scatter. Возвращает путь к png или None если matplotlib недоступен."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib не установлен — пропускаю plots. pip install matplotlib")
        return None

    deltas_correct = [r["delta_ensemble"] for r in records if r.get("judge_correct") is True]
    deltas_wrong   = [r["delta_ensemble"] for r in records if r.get("judge_correct") is False]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # (1) Гистограмма δ, раскраска по judge-correctness
    ax1 = axes[0]
    bins = 12
    if deltas_correct:
        ax1.hist(deltas_correct, bins=bins, alpha=0.65, label=f"judge correct (n={len(deltas_correct)})",
                 color="#2a9d8f", edgecolor="white")
    if deltas_wrong:
        ax1.hist(deltas_wrong, bins=bins, alpha=0.65, label=f"judge wrong (n={len(deltas_wrong)})",
                 color="#e76f51", edgecolor="white")
    ax1.set_xlabel("δ = sim(q, d⁻) / sim(q, d⁺)")
    ax1.set_ylabel("count")
    ax1.set_title("CE δ на 31 контрфакте\nраскраска по correctness LLM-judge")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.axvline(1.0, color="grey", linestyle=":", linewidth=1, label="δ=1.0 (паритет)")

    # (2) Scatter: δ per-encoder (если ровно два)
    ax2 = axes[1]
    enc_names = list(records[0]["delta_per_enc"].keys()) if records else []
    if len(enc_names) >= 2:
        x = [r["delta_per_enc"][enc_names[0]] for r in records]
        y = [r["delta_per_enc"][enc_names[1]] for r in records]
        colors = [
            "#e76f51" if r.get("judge_correct") is False
            else "#2a9d8f" if r.get("judge_correct") is True
            else "#888888"
            for r in records
        ]
        ax2.scatter(x, y, c=colors, alpha=0.75, s=60, edgecolors="white", linewidths=1)
        # Диагональ
        lo = min(min(x), min(y))
        hi = max(max(x), max(y))
        ax2.plot([lo, hi], [lo, hi], color="grey", linestyle=":", linewidth=1)
        ax2.set_xlabel(f"δ ({enc_names[0].split('/')[-1]})")
        ax2.set_ylabel(f"δ ({enc_names[1].split('/')[-1]})")
        ax2.set_title("Согласованность δ между энкодерами")
    else:
        ax2.text(0.5, 0.5, "Нужно ≥2 энкодера", ha="center", va="center")

    plt.tight_layout()
    path = output_dir / "ce_distribution.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info("Графики сохранены: %s", path)
    return path


# ──────────────────────────────────────────────────────────────────────
# Threshold-анализ
# ──────────────────────────────────────────────────────────────────────

def threshold_sweep(records: list[dict]) -> list[dict]:
    """
    Для серии порогов τ ∈ [0.5..1.1] считаем confusion matrix:
      - Принимаем CF как «hard negative» если δ < τ (CE считает: отличим от d⁺).
      - "Ground truth": судья сказал "Нет" (correct=True для CF) → действительно негатив.

    Это позволяет понять: можем ли мы заменить LLM-judge порогом по δ?
    """
    # Только CF, где есть judge verdict
    valid = [r for r in records if r.get("judge_correct") is not None]

    rows = []
    for tau in [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.95, 1.00, 1.05, 1.10]:
        # CE accept (δ < τ → CE говорит «достаточно отличим, годится как негатив»)
        tp = sum(1 for r in valid if r["delta_ensemble"] < tau and r["judge_correct"])
        fp = sum(1 for r in valid if r["delta_ensemble"] < tau and not r["judge_correct"])
        tn = sum(1 for r in valid if r["delta_ensemble"] >= tau and not r["judge_correct"])
        fn = sum(1 for r in valid if r["delta_ensemble"] >= tau and r["judge_correct"])
        n = tp + fp + tn + fn
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        rows.append({
            "tau": tau, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "n": n,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
        })
    return rows


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs/default.yaml"))
    ap.add_argument("--counterfactuals",
                    default=str(PROJECT_ROOT / "outputs/sanity/counterfactuals.jsonl"))
    ap.add_argument("--judge",
                    default=str(PROJECT_ROOT / "outputs/sanity/judge.jsonl"))
    ap.add_argument("--output-dir",
                    default=str(PROJECT_ROOT / "outputs/sanity/ce_experiment"))
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfs = load_jsonl(Path(args.counterfactuals))
    verdicts = load_jsonl(Path(args.judge))
    log.info("Загружено: %d CF, %d judge verdicts", len(cfs), len(verdicts))

    # Энкодеры — это может занять время на первом запуске (скачивание e5-large ~2 GB,
    # bge-m3 ~2 GB в W:\huggingface_cache). После — кэш.
    log.info("Загружаю ансамбль энкодеров (e5-large + bge-m3)…")
    pool = EncoderPool.from_config(cfg["encoders"])
    log.info("Готово: %d энкодеров", len(pool.encoders))

    # Матчинг CF ↔ judge
    matched_verdicts = match_judge_verdicts(cfs, verdicts)

    # Подсчёт δ
    records = compute_deltas(pool, cfs, matched_verdicts)

    # Анализ
    summary = analyze(records)
    thresholds = threshold_sweep(records)

    # Сохранение
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_jsonl(out_dir / "ce_deltas.jsonl", records)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "threshold_sweep.json").write_text(
        json.dumps(thresholds, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_path = make_plots(records, out_dir)

    # Печать сводки
    print("\n" + "=" * 70)
    print("CE EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"N counterfactuals analyzed: {len(records)}")
    print()
    print("Overall δ distribution:")
    o = summary["overall"]
    print(f"  mean={o['mean']:.3f}  median={o['median']:.3f}  stdev={o['stdev']:.3f}  range=[{o['min']:.3f}, {o['max']:.3f}]")
    print()
    print("δ split by judge correctness:")
    if summary["judge_correct_subgroup"]["n"]:
        c = summary["judge_correct_subgroup"]
        print(f"  judge CORRECT (said 'Нет'):  n={c['n']}  mean={c['mean']:.3f}  median={c['median']:.3f}")
    if summary["judge_wrong_subgroup"]["n"]:
        w = summary["judge_wrong_subgroup"]
        print(f"  judge WRONG   (said 'Да'):   n={w['n']}  mean={w['mean']:.3f}  median={w['median']:.3f}")
    print()
    if (summary["judge_correct_subgroup"]["n"] and summary["judge_wrong_subgroup"]["n"]):
        diff = summary["judge_wrong_subgroup"]["mean"] - summary["judge_correct_subgroup"]["mean"]
        print(f"  → разница mean(δ) для wrong − correct = {diff:+.3f}")
        if diff > 0.05:
            print("    Положительный сдвиг: где LLM ошибается — там CE видит более похожие CF на d⁺.")
            print("    CE как фильтр имеет смысл (отбрасывать CF с δ > threshold).")
        elif diff > 0:
            print("    Слабый сигнал — CE даёт намёк, но граница не разделяет чётко.")
        else:
            print("    Сигнала нет — CE не разделяет правильные/неправильные CF.")
            print("    Это значит: одной CE-метрики недостаточно, нужен LLM-judge или другой подход.")
    print()
    print("Encoder correlation (Pearson):")
    print(f"  {summary['encoder_correlation_pearson']}  (1.0 = идеальная согласованность)")
    print()
    print("δ by mutation type:")
    for t, s in sorted(summary["by_mutation_type"].items()):
        if s["n"]:
            print(f"  {t:8s}: n={s['n']}  mean={s['mean']:.3f}  range=[{s['min']:.3f}, {s['max']:.3f}]")
    print()
    print("Threshold sweep (δ < τ → принимаем CF):")
    print(f"  {'τ':>5}  {'TP':>3} {'FP':>3} {'TN':>3} {'FN':>3}  {'precis':>6} {'recall':>6} {'f1':>5}")
    for row in thresholds:
        print(f"  {row['tau']:>5.2f}  {row['tp']:>3} {row['fp']:>3} {row['tn']:>3} {row['fn']:>3}  "
              f"{row['precision']:>6.3f} {row['recall']:>6.3f} {row['f1']:>5.3f}")
    print()
    print(f"Files:")
    print(f"  {out_dir / 'ce_deltas.jsonl'}")
    print(f"  {out_dir / 'summary.json'}")
    print(f"  {out_dir / 'threshold_sweep.json'}")
    if plot_path:
        print(f"  {plot_path}")


if __name__ == "__main__":
    main()
