#!/usr/bin/env python3
"""
Phase 0: Sanity check Qwen2.5-7B на рукотворных 15 примерах.

Что измеряем:
  1. Fact extraction:
     - доля валидного JSON
     - hallucination rate (факт не verbatim в d⁺ И не морф-вариант)
     - morphological_variant rate (не verbatim, но проходит fuzzy)
     - recall vs expected_mutable_facts (через fuzzy_in, по ВСЕМ извлечённым фактам)
     - распределение criticality
  2. Counterfactual mutation:
     - self-similarity rate (LLM вернула оригинал)
     - length preservation ratio
  3. LLM judge:
     - precision на (q, d⁺)         → ждём "Да"
     - recall  на (q, unrelated_d)  → ждём "Нет"
     - на (q, d⁻)                    → ждём "Нет"

Выход:
  outputs/sanity/
    ├── facts.jsonl         — все извлечённые факты с полем match_status
    ├── counterfactuals.jsonl
    ├── judge.jsonl
    ├── metrics.json
    └── report.md           — для человеческого просмотра

Запуск:
  python scripts/00_sanity_qwen.py --config configs/default.yaml

Версия: Phase 0.5 fix — recall считается по all_facts (не valid_facts),
                       морф-варианты выделены отдельно от честных галлюцинаций.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, stdev

import yaml

# Корень проекта = родитель папки scripts/
# Это работает независимо от того, откуда запускается скрипт:
#   python scripts/00_sanity_qwen.py          (из корня)
#   python 00_sanity_qwen.py                  (из scripts/)
#   W:\Jupyter\NIR_2\scripts\00_sanity_qwen.py  (абсолютный путь)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.llm_client import LLMConfig, make_client, extract_json, parse_yes_no, parse_cot_verdict  # noqa: E402
from src.prompts import (  # noqa: E402
    build_fact_extraction_messages,
    build_counterfactual_messages,
    build_judge_messages,
    build_judge_cot_messages,
)
from src.text_utils import fuzzy_in, morph_available  # noqa: E402

log = logging.getLogger("sanity")


# ──────────────────────────────────────────────────────────────────────
# Утилиты
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


# fuzzy_in импортируется из src.text_utils — теперь с лемматизацией через pymorphy3.
# Это чинит recall на русской морфологии: "Льва Николаевича Толстого" ↔ "Лев Толстой".


# ──────────────────────────────────────────────────────────────────────
# Этап 1: Fact extraction
# ──────────────────────────────────────────────────────────────────────
#
# АРХИТЕКТУРНАЯ ЛОГИКА (важно):
#
#   У извлечённого факта три возможных статуса по отношению к d⁺:
#
#     1. "verbatim"             — точная подстрока d⁺ (strict in-check).
#                                 Только такие факты идут в CN-промпт, т.к. LLM
#                                 должна физически найти и заменить подстроку.
#     2. "morphological_variant"— не подстрока, но fuzzy_in() == True.
#                                 Это правильно извлечённый факт, просто в другой
#                                 морфологической форме. НЕ галлюцинация.
#                                 Не годится для CN-промпта в текущем виде.
#     3. "hallucinated"         — не подстрока И не проходит fuzzy_in.
#                                 Настоящая выдумка, не присутствует в d⁺.
#
#   Recall vs expected_mutable_facts должен считаться по ВСЕМ извлечённым фактам
#   (категории 1 и 2), потому что это метрика поведения модели — извлекла она
#   нужное содержание или нет. Старый код считал только по категории 1, что
#   занижало recall на русской морфологии.

def _classify_fact(fact_text: str, d_plus: str) -> str:
    """Вернуть один из 'verbatim' / 'morphological_variant' / 'hallucinated'."""
    if fact_text in d_plus:
        return "verbatim"
    if fuzzy_in(fact_text, d_plus):
        return "morphological_variant"
    return "hallucinated"


def run_fact_extraction(llm, examples: list[dict]) -> list[dict]:
    """Возвращает по записи на каждый пример с полной диагностикой."""
    log.info("=== Этап 1: Fact extraction (%d примеров) ===", len(examples))

    try:
        from tqdm import tqdm
        pbar = tqdm(total=len(examples), desc="Fact extraction", unit="ex")
    except ImportError:
        pbar = None

    results = []
    lost_for_cn: list[str] = []  # qid, у которых 0 verbatim-фактов → выпадают из CN

    for ex in examples:
        msgs = build_fact_extraction_messages(ex["query"], ex["d_plus"])
        raw = llm.generate([msgs])[0]

        parsed = extract_json(raw)
        valid_json = isinstance(parsed, list)

        all_facts: list[dict] = []
        valid_facts: list[dict] = []        # verbatim в d⁺ — годятся для CN
        morph_variant_facts: list[dict] = []  # морф-вариант — НЕ галлюцинация
        hallucinated_facts: list[dict] = []   # настоящая галлюцинация

        if valid_json:
            for item in parsed:
                if not isinstance(item, dict) or "text" not in item:
                    continue
                fact = {
                    "text": str(item.get("text", "")).strip(),
                    "type": str(item.get("type", "")).strip(),
                    "criticality": int(item.get("criticality", 0)),
                }
                status = _classify_fact(fact["text"], ex["d_plus"])
                fact["match_status"] = status
                all_facts.append(fact)

                if status == "verbatim":
                    valid_facts.append(fact)
                elif status == "morphological_variant":
                    morph_variant_facts.append(fact)
                else:  # "hallucinated"
                    hallucinated_facts.append(fact)

        # Recall теперь по ВСЕМ извлечённым фактам (verbatim + морф-варианты).
        # Тем самым он отражает способность модели вытащить факт по содержанию,
        # а не способность пройти строгий verbatim-чек.
        expected = ex["expected_mutable_facts"]
        covered = [
            e for e in expected
            if any(fuzzy_in(e, f["text"]) for f in all_facts)
        ]
        recall = len(covered) / len(expected) if expected else 0.0

        # Каскадная потеря: 0 verbatim-фактов → CN-пайплайн пропустит пример,
        # хотя модель что-то извлекла. Логируем явно.
        cn_lost = bool(all_facts) and not valid_facts
        if cn_lost:
            lost_for_cn.append(ex["qid"])

        results.append({
            "qid": ex["qid"],
            "query": ex["query"],
            "d_plus": ex["d_plus"],
            "expected_facts": expected,
            "raw_output": raw,
            "valid_json": valid_json,
            "all_facts": all_facts,
            "valid_facts": valid_facts,
            "morph_variant_facts": morph_variant_facts,
            "hallucinated_facts": hallucinated_facts,
            "covered_expected": covered,
            "recall_vs_expected": recall,
            "cn_lost": cn_lost,
        })
        log.info(
            "  %s: json=%s facts=%d verbatim=%d morph=%d hallu=%d recall=%.2f%s",
            ex["qid"], valid_json, len(all_facts),
            len(valid_facts), len(morph_variant_facts),
            len(hallucinated_facts), recall,
            "  [LOST FOR CN]" if cn_lost else "",
        )
        if pbar:
            pbar.set_postfix(recall=f"{recall:.0%}",
                             verbatim=len(valid_facts),
                             morph=len(morph_variant_facts))
            pbar.update(1)

    if pbar:
        pbar.close()

    if lost_for_cn:
        log.warning(
            "⚠️  %d пример(ов) выпали из CN-пайплайна (0 verbatim-фактов, но факты извлечены): %s",
            len(lost_for_cn), ", ".join(lost_for_cn),
        )
        log.warning(
            "   Это означает: модель достала факт в другой морф-форме, CN-промпт не сможет его найти и заменить."
        )

    return results


# ──────────────────────────────────────────────────────────────────────
# Этап 2: Counterfactual mutation для top-K фактов
# ──────────────────────────────────────────────────────────────────────

def run_counterfactuals(
    llm, fact_results: list[dict], k_facts: int = 3
) -> list[dict]:
    log.info("=== Этап 2: Counterfactual mutation (top-%d по criticality) ===", k_facts)

    pairs, batch_msgs = [], []
    for fr in fact_results:
        # CN использует ТОЛЬКО verbatim-факты — это правильно, иначе LLM не сможет
        # физически найти и подменить подстроку в d⁺.
        top = sorted(fr["valid_facts"], key=lambda f: -f["criticality"])[:k_facts]
        for fact in top:
            pairs.append((fr, fact))
            batch_msgs.append(build_counterfactual_messages(
                query=fr["query"], d_plus=fr["d_plus"],
                fact_text=fact["text"], fact_type=fact["type"],
            ))

    if not batch_msgs:
        log.warning("Нет фактов для мутации — пропускаем этап 2.")
        return []

    try:
        from tqdm import tqdm
        iter_pairs = tqdm(zip(pairs, batch_msgs), total=len(pairs),
                          desc="Counterfactual", unit="fact")
    except ImportError:
        iter_pairs = zip(pairs, batch_msgs)

    cf_results = []
    for (fr, fact), msgs in iter_pairs:
        raw = llm.generate([msgs])[0]
        d_plus = fr["d_plus"]
        d_minus = raw.strip()
        if d_minus.startswith('"') and d_minus.endswith('"') and len(d_minus) > 2:
            d_minus = d_minus[1:-1].strip()

        is_self = d_minus == d_plus.strip()
        length_ratio = len(d_minus) / len(d_plus) if d_plus else 0.0
        fact_still_present = fact["text"] in d_minus

        cf_results.append({
            "qid": fr["qid"],
            "query": fr["query"],
            "d_plus": d_plus,
            "fact_text": fact["text"],
            "fact_type": fact["type"],
            "fact_criticality": fact["criticality"],
            "d_minus": d_minus,
            "raw_output": raw,
            "is_self_copy": is_self,
            "length_ratio": length_ratio,
            "fact_still_present": fact_still_present,
        })
        log.info(
            "  %s [%s]: self=%s len_ratio=%.2f fact_still_in_d-=%s",
            fr["qid"], fact["text"][:30], is_self,
            length_ratio, fact_still_present,
        )
    return cf_results


# ──────────────────────────────────────────────────────────────────────
# Этап 3: LLM judge
# ──────────────────────────────────────────────────────────────────────

def run_judge(
    llm, examples: list[dict], cf_results: list[dict], llm_cfg: LLMConfig
) -> dict:
    mode = (llm_cfg.judge_mode or "cot").lower()
    if mode not in ("cot", "fast"):
        log.warning("Неизвестный judge_mode=%r — fallback на 'cot'", mode)
        mode = "cot"
    log.info("=== Этап 3: LLM judge (mode=%s) ===", mode)

    # Готовим все пары для одного батч-прогона
    pairs, kinds, qids = [], [], []
    for ex in examples:
        pairs.append((ex["query"], ex["d_plus"]))
        kinds.append("positive_control")
        qids.append(ex["qid"])

        pairs.append((ex["query"], ex["unrelated_d"]))
        kinds.append("negative_control")
        qids.append(ex["qid"])

    for cf in cf_results:
        if cf["is_self_copy"]:
            continue  # бессмысленно судить копию d⁺
        pairs.append((cf["query"], cf["d_minus"]))
        kinds.append("counterfactual")
        qids.append(cf["qid"])

    # Сборка сообщений и лимит токенов под режим
    if mode == "cot":
        build = build_judge_cot_messages
        max_toks = llm_cfg.judge_cot_max_new_tokens
        parser = parse_cot_verdict           # → (verdict, reasoning)
    else:
        build = build_judge_messages
        max_toks = llm_cfg.judge_max_new_tokens
        parser = lambda raw: (parse_yes_no(raw), "")  # noqa: E731

    batch_msgs = [build(q, d) for q, d in pairs]
    try:
        from tqdm import tqdm
        iter_batch = tqdm(zip(pairs, kinds, qids, batch_msgs),
                          total=len(pairs), desc=f"Judge[{mode}]", unit="pair")
    except ImportError:
        iter_batch = zip(pairs, kinds, qids, batch_msgs)

    verdicts = []
    for (q, d), kind, qid, msgs in iter_batch:
        raw = llm.generate(
            [msgs],
            temperature=llm_cfg.judge_temperature,
            max_new_tokens=max_toks,
        )[0]
        v, reasoning = parser(raw)
        expected = "yes" if kind == "positive_control" else "no"
        verdicts.append({
            "qid": qid,
            "kind": kind,
            "judge_mode": mode,
            "query": q,
            "doc": d,
            "raw_output": raw,
            "reasoning": reasoning,
            "verdict": v,
            "expected": expected,
            "correct": v == expected,
        })
    return {"verdicts": verdicts, "mode": mode}


# ──────────────────────────────────────────────────────────────────────
# Метрики
# ──────────────────────────────────────────────────────────────────────

def compute_metrics(facts, cfs, judge):
    n = len(facts)
    # Fact extraction
    json_ok_rate = sum(f["valid_json"] for f in facts) / n if n else 0.0

    # facts_per_doc — считаем по verbatim-фактам (это то, что реально пойдёт в CN)
    facts_per_doc_verbatim = [len(f["valid_facts"]) for f in facts]
    facts_per_doc_total    = [len(f["all_facts"])   for f in facts]

    total_all   = max(1, sum(len(f["all_facts"]) for f in facts))
    total_hallu = sum(len(f["hallucinated_facts"]) for f in facts)
    total_morph = sum(len(f.get("morph_variant_facts", [])) for f in facts)

    # Истинная hallucination rate — только настоящие выдумки, без морф-вариантов
    hallu_rate = total_hallu / total_all
    # Отдельно — доля морф-вариантов (показатель русского NLP-шума, а не качества LLM)
    morph_rate = total_morph / total_all
    # Старая (strict) метрика для сравнения с предыдущими прогонами
    strict_hallu_rate = (total_hallu + total_morph) / total_all

    recalls = [f["recall_vs_expected"] for f in facts]
    # criticality считаем по фактам, годным для CN
    crits = [it["criticality"] for f in facts for it in f["valid_facts"]]

    # Каскадные потери для CN
    lost_for_cn = [f["qid"] for f in facts if f.get("cn_lost")]

    # Counterfactual
    cf_self = (
        sum(c["is_self_copy"] for c in cfs) / len(cfs) if cfs else 0.0
    )
    cf_len_ratios = [c["length_ratio"] for c in cfs]
    fact_leak = (
        sum(c["fact_still_present"] for c in cfs) / len(cfs) if cfs else 0.0
    )

    # Judge — точность по типам
    by_kind = {"positive_control": [], "negative_control": [], "counterfactual": []}
    for v in judge["verdicts"]:
        by_kind[v["kind"]].append(v["correct"])
    j_pos = mean(by_kind["positive_control"]) if by_kind["positive_control"] else None
    j_neg = mean(by_kind["negative_control"]) if by_kind["negative_control"] else None
    j_cf  = mean(by_kind["counterfactual"])    if by_kind["counterfactual"]    else None

    return {
        "n_examples": n,
        "fact_extraction": {
            "json_parse_rate": json_ok_rate,
            "facts_per_doc_verbatim_mean": mean(facts_per_doc_verbatim) if facts_per_doc_verbatim else 0,
            "facts_per_doc_verbatim_stdev": stdev(facts_per_doc_verbatim) if len(facts_per_doc_verbatim) > 1 else 0,
            "facts_per_doc_total_mean": mean(facts_per_doc_total) if facts_per_doc_total else 0,
            "hallucination_rate": hallu_rate,
            "morph_variant_rate": morph_rate,
            "strict_hallucination_rate_legacy": strict_hallu_rate,
            "recall_vs_expected_mean": mean(recalls) if recalls else 0,
            "n_examples_lost_for_cn": len(lost_for_cn),
            "examples_lost_for_cn": lost_for_cn,
            "criticality_mean": mean(crits) if crits else 0,
            "criticality_dist": dict(Counter(crits)),
        },
        "counterfactual": {
            "n_generated": len(cfs),
            "self_copy_rate": cf_self,
            "length_ratio_mean": mean(cf_len_ratios) if cf_len_ratios else 0,
            "length_ratio_stdev": stdev(cf_len_ratios) if len(cf_len_ratios) > 1 else 0,
            "fact_still_present_rate": fact_leak,  # хотим близко к 0
        },
        "judge": {
            "accuracy_on_positive_q_dplus": j_pos,
            "accuracy_on_negative_q_unrelated": j_neg,
            "accuracy_on_counterfactual_q_dminus": j_cf,
        },
    }


# ──────────────────────────────────────────────────────────────────────
# Markdown отчёт
# ──────────────────────────────────────────────────────────────────────

def write_report(facts, cfs, judge, metrics, output_dir: Path):
    lines = ["# Phase 0 — Sanity Report", ""]
    lines.append(f"- Режим LLM-судьи: **{judge.get('mode', 'fast')}**")
    lines.append(f"- Лемматизация (pymorphy3): **{'доступна' if morph_available() else 'НЕТ — fuzzy_in работает в substring-режиме'}**")
    lines.append("")

    # Сводка
    lines.append("## Сводные метрики\n")
    fe = metrics["fact_extraction"]
    cm = metrics["counterfactual"]
    jm = metrics["judge"]
    lines.append("### Fact extraction")
    lines.append(f"- JSON parse rate: **{fe['json_parse_rate']:.1%}**")
    lines.append(f"- Verbatim-фактов на документ: **{fe['facts_per_doc_verbatim_mean']:.1f} ± {fe['facts_per_doc_verbatim_stdev']:.1f}** (то, что идёт в CN)")
    lines.append(f"- Всего фактов на документ: **{fe['facts_per_doc_total_mean']:.1f}** (verbatim + морф-варианты + галлюцинации)")
    lines.append(f"- Hallucination rate (честная, без морф-вариантов): **{fe['hallucination_rate']:.1%}**")
    lines.append(f"- Morphological variant rate: **{fe['morph_variant_rate']:.1%}** *(факт извлечён правильно, но не verbatim в d⁺)*")
    lines.append(f"- Strict-hallu rate (legacy, для сравнения с предыдущими прогонами): **{fe['strict_hallucination_rate_legacy']:.1%}**")
    lines.append(f"- Recall vs expected_mutable_facts (через fuzzy_in по ВСЕМ извлечённым): **{fe['recall_vs_expected_mean']:.1%}**")
    if fe["examples_lost_for_cn"]:
        lines.append(f"- ⚠️ Примеров потеряно для CN (0 verbatim, но факты были): **{fe['n_examples_lost_for_cn']}** — `{fe['examples_lost_for_cn']}`")
    lines.append(f"- Средняя criticality (по verbatim-фактам): **{fe['criticality_mean']:.2f}**, распределение: `{fe['criticality_dist']}`")
    lines.append("")
    lines.append("### Counterfactual")
    lines.append(f"- Сгенерировано: **{cm['n_generated']}**")
    lines.append(f"- Self-copy rate (вернули оригинал): **{cm['self_copy_rate']:.1%}** *(хотим 0)*")
    lines.append(f"- Length ratio (len(d⁻)/len(d⁺)): **{cm['length_ratio_mean']:.2f} ± {cm['length_ratio_stdev']:.2f}** *(хотим ~1.0)*")
    lines.append(f"- Старый факт ещё в d⁻: **{cm['fact_still_present_rate']:.1%}** *(хотим 0)*")
    lines.append("")
    lines.append("### Judge")
    lines.append(f"- Acc на (q, d⁺) → ожидание «Да»: **{jm['accuracy_on_positive_q_dplus']:.1%}** *(хотим 1.0)*")
    lines.append(f"- Acc на (q, unrelated) → ожидание «Нет»: **{jm['accuracy_on_negative_q_unrelated']:.1%}** *(хотим 1.0)*")
    lines.append(f"- Acc на (q, d⁻) → ожидание «Нет»: **{jm['accuracy_on_counterfactual_q_dminus']:.1%}**")
    lines.append("")

    # Per-example
    lines.append("## Подробно по примерам")
    lines.append("")
    cf_by_qid = {}
    for c in cfs:
        cf_by_qid.setdefault(c["qid"], []).append(c)
    judge_by_qid = {}
    for v in judge["verdicts"]:
        judge_by_qid.setdefault(v["qid"], []).append(v)

    for f in facts:
        qid = f["qid"]
        lines.append(f"### `{qid}`: {f['query']}")
        lines.append("")
        lines.append(f"**d⁺**: {f['d_plus']}")
        lines.append("")
        lines.append(f"**Ожидаемые факты**: {f['expected_facts']}")
        lines.append(f"**Покрыто**: {f['covered_expected']} (recall={f['recall_vs_expected']:.0%})")
        if f.get("cn_lost"):
            lines.append("⚠️  **CN-пайплайн пропустит этот пример** — нет verbatim-фактов, есть только морф-варианты.")
        lines.append("")
        if not f["valid_json"]:
            lines.append("⚠️  **JSON не распарсился**. Raw:")
            lines.append("```")
            lines.append(f["raw_output"][:500])
            lines.append("```")

        # Все извлечённые факты единой таблицей с явным статусом
        if f["all_facts"]:
            lines.append("**Извлечённые факты:**")
            lines.append("")
            lines.append("| text | type | crit | статус |")
            lines.append("|---|---|---|---|")
            status_icon = {
                "verbatim": "✓ verbatim",
                "morphological_variant": "≈ морф-вариант",
                "hallucinated": "✗ галлюцинация",
            }
            for ff in f["all_facts"]:
                tx = ff["text"].replace("|", "\\|")
                ty = ff["type"].replace("|", "\\|")
                st = status_icon.get(ff.get("match_status", "?"), "?")
                lines.append(f"| {tx} | {ty} | {ff['criticality']} | {st} |")
            lines.append("")
        if f["hallucinated_facts"]:
            lines.append("⚠️  **Hallucinated (не подстрока И не морф-вариант d⁺):**")
            for ff in f["hallucinated_facts"]:
                lines.append(f"- `{ff['text']}` (type={ff['type']}, crit={ff['criticality']})")
            lines.append("")

        # Контрфакты
        for c in cf_by_qid.get(qid, []):
            lines.append(f"**Контрфакт по факту**: `{c['fact_text']}` ({c['fact_type']}, crit={c['fact_criticality']})")
            lines.append("")
            lines.append(f"_d⁻_: {c['d_minus']}")
            lines.append("")
            flags = []
            if c["is_self_copy"]: flags.append("❌ self-copy")
            if c["fact_still_present"]: flags.append("⚠️ старый факт ещё внутри")
            if not (0.7 <= c["length_ratio"] <= 1.4):
                flags.append(f"⚠️ длина {c['length_ratio']:.2f}x")
            lines.append(f"_len ratio={c['length_ratio']:.2f}_  {' | '.join(flags) if flags else '✓'}")
            lines.append("")

        # Verdicts
        verdicts = judge_by_qid.get(qid, [])
        if verdicts:
            has_reasoning = any(v.get("reasoning") for v in verdicts)
            lines.append("**Judge verdicts:**")
            lines.append("")
            if has_reasoning:
                lines.append("| тип | вердикт | ожидание | OK | reasoning |")
                lines.append("|---|---|---|---|---|")
                for v in verdicts:
                    ok = "✓" if v["correct"] else "❌"
                    r = (v.get("reasoning") or "").replace("\n", " ").replace("|", "\\|")
                    if len(r) > 300:
                        r = r[:300] + "…"
                    lines.append(f"| {v['kind']} | {v['verdict']} | {v['expected']} | {ok} | {r} |")
            else:
                lines.append("| тип | вердикт | ожидание | OK |")
                lines.append("|---|---|---|---|")
                for v in verdicts:
                    ok = "✓" if v["correct"] else "❌"
                    lines.append(f"| {v['kind']} | {v['verdict']} | {v['expected']} | {ok} |")
            lines.append("")
        lines.append("---")
        lines.append("")

    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Отчёт записан в %s", report_path)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",      default=str(PROJECT_ROOT / "configs/default.yaml"))
    ap.add_argument("--examples",    default=str(PROJECT_ROOT / "data/sanity_examples.jsonl"))
    ap.add_argument("--output-dir",  default=str(PROJECT_ROOT / "outputs/sanity"))
    ap.add_argument("--limit", type=int, default=None,
                    help="ограничить число примеров (для отладки)")
    ap.add_argument("--k-facts", type=int, default=3,
                    help="сколько top-критичных фактов мутировать на пример")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    examples = load_jsonl(Path(args.examples))
    if args.limit:
        examples = examples[: args.limit]
    log.info("Загружено %d примеров", len(examples))

    llm_cfg = LLMConfig.from_dict(cfg["llm"])
    log.info("LLM: %s (backend=%s)", llm_cfg.model_name, llm_cfg.backend)
    log.info("Judge mode: %s", llm_cfg.judge_mode)
    log.info("pymorphy3: %s", "доступен" if morph_available()
             else "НЕ установлен — fuzzy_in работает в substring-режиме")
    llm = make_client(llm_cfg)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    facts = run_fact_extraction(llm, examples)
    save_jsonl(out_dir / "facts.jsonl", facts)

    cfs = run_counterfactuals(llm, facts, k_facts=args.k_facts)
    save_jsonl(out_dir / "counterfactuals.jsonl", cfs)

    judge = run_judge(llm, examples, cfs, llm_cfg)
    save_jsonl(out_dir / "judge.jsonl", judge["verdicts"])

    metrics = compute_metrics(facts, cfs, judge)
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(facts, cfs, judge, metrics, out_dir)

    # Краткая сводка в консоль
    print("\n" + "=" * 60)
    print("PHASE 0 SUMMARY")
    print("=" * 60)
    fe = metrics["fact_extraction"]
    cm = metrics["counterfactual"]
    jm = metrics["judge"]
    print(f"Fact extraction:")
    print(f"  JSON parse:        {fe['json_parse_rate']:.1%}")
    print(f"  Verbatim/doc:      {fe['facts_per_doc_verbatim_mean']:.1f} ± {fe['facts_per_doc_verbatim_stdev']:.1f}")
    print(f"  Total/doc:         {fe['facts_per_doc_total_mean']:.1f}")
    print(f"  Hallucination:     {fe['hallucination_rate']:.1%}  (честная)")
    print(f"  Morph-variant:     {fe['morph_variant_rate']:.1%}  (правильные факты, не verbatim)")
    print(f"  Recall expected:   {fe['recall_vs_expected_mean']:.1%}")
    if fe["examples_lost_for_cn"]:
        print(f"  Lost for CN:       {fe['n_examples_lost_for_cn']}  -> {fe['examples_lost_for_cn']}")
    print(f"Counterfactual:")
    print(f"  Self-copy:         {cm['self_copy_rate']:.1%}  (want 0)")
    print(f"  Length ratio:      {cm['length_ratio_mean']:.2f}±{cm['length_ratio_stdev']:.2f}  (want ~1.0)")
    print(f"  Fact still in:     {cm['fact_still_present_rate']:.1%}  (want 0)")
    print(f"Judge:")
    print(f"  (q,d⁺)→Да:         {jm['accuracy_on_positive_q_dplus']:.1%}  (want 1.0)")
    print(f"  (q,unrel)→Нет:     {jm['accuracy_on_negative_q_unrelated']:.1%}  (want 1.0)")
    print(f"  (q,d⁻)→Нет:        {jm['accuracy_on_counterfactual_q_dminus']:.1%}")
    print(f"\nПодробный отчёт: {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
