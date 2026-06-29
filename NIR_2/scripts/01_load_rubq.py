#!/usr/bin/env python3
"""
Phase 1: Загрузка RuBQ 2.0 и подготовка пар (запрос, d⁺) для пайплайна.

Что делаем:
  1. Загружаем датасет d0rj/RuBQ_2.0 (вопросы) и d0rj/RuBQ_2.0-paragraphs (корпус).
  2. Индексируем параграфы по uid.
  3. Для каждого вопроса берём первый параграф из paragraphs_uids.with_answer
     как d⁺ (если такой есть). Фильтруем по разумным длинам.
  4. Сохраняем:
        data/rubq/rubq_full.jsonl   — все валидные пары (qid, query, d_plus, meta)
        data/rubq/rubq_dev50.jsonl  — детерминированный сэмпл из 50 для dev-set
        data/rubq/stats.json        — статистика
  5. Выводим в консоль сводную статистику.

Структура источников (на 2026-06):
  Questions split:
    - test: 2330 rows
    - dev:  580  rows
  Поля вопроса:
    - uid, question_text, query, answer_text
    - paragraphs_uids: {"all_related": [...], "with_answer": [...]}
    - tags: ["1-hop"] | ["multi-constraint"] | ...
    - RuBQ_version: "1.0" | "2.0"
  Поля параграфа:
    - uid, ru_wiki_pageid, text

Запуск (Windows):
  set HF_HOME=W:\\huggingface_cache
  python scripts/01_load_rubq.py
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median, quantiles, stdev

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

log = logging.getLogger("load_rubq")


# Имена датасетов на HuggingFace
QUESTIONS_REPO = "d0rj/RuBQ_2.0"
PARAGRAPHS_REPO = "d0rj/RuBQ_2.0-paragraphs"


# ──────────────────────────────────────────────────────────────────────
# Загрузка
# ──────────────────────────────────────────────────────────────────────

def load_datasets(use_splits: list[str], hf_cache: str | None):
    """
    Загружает оба датасета и возвращает (questions_records, paragraphs_by_uid).

    questions_records — плоский список dict из выбранных сплитов.
    paragraphs_by_uid — dict[int → str].
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit(
            "Не установлен пакет `datasets`. Сделай `pip install datasets`."
        ) from e

    if hf_cache:
        os.environ["HF_HOME"] = hf_cache

    log.info("Загружаем вопросы: %s (splits=%s)", QUESTIONS_REPO, use_splits)
    q_records: list[dict] = []
    for split in use_splits:
        ds = load_dataset(QUESTIONS_REPO, split=split)
        log.info("  %s: %d строк", split, len(ds))
        for row in ds:
            row["_source_split"] = split
            q_records.append(row)

    log.info("Загружаем корпус параграфов: %s", PARAGRAPHS_REPO)
    p_ds = load_dataset(PARAGRAPHS_REPO, split="paragraphs")
    log.info("  paragraphs: %d строк", len(p_ds))
    paragraphs_by_uid: dict[int, str] = {}
    # row['uid'] — int, row['text'] — str
    for row in p_ds:
        uid = int(row["uid"])
        txt = row["text"]
        if isinstance(txt, str) and txt.strip():
            paragraphs_by_uid[uid] = txt

    log.info("В индексе параграфов: %d записей", len(paragraphs_by_uid))
    return q_records, paragraphs_by_uid


# ──────────────────────────────────────────────────────────────────────
# Сборка (query, d⁺) пар
# ──────────────────────────────────────────────────────────────────────

def _extract_with_answer_uids(row: dict) -> list[int]:
    """
    paragraphs_uids в датасете — dict с ключами 'all_related' и 'with_answer'
    (списки int). Иногда они могут быть пустыми или отсутствовать.
    """
    p = row.get("paragraphs_uids") or {}
    wa = p.get("with_answer") or []
    return [int(x) for x in wa if x is not None]


def build_pairs(
    q_records: list[dict],
    paragraphs_by_uid: dict[int, str],
    *,
    min_doc_chars: int = 80,
    max_doc_chars: int = 2500,
    min_question_chars: int = 8,
    max_question_chars: int = 250,
) -> tuple[list[dict], dict]:
    """
    Возвращает (pairs, drop_stats).

    pairs — список {qid, query, d_plus, meta}.
    drop_stats — Counter, почему примеры отбрасывались.
    """
    pairs: list[dict] = []
    drops: Counter = Counter()

    for row in q_records:
        uid = row.get("uid")
        question = (row.get("question_text") or "").strip()
        answer = (row.get("answer_text") or "").strip()
        tags = list(row.get("tags") or [])

        if not question:
            drops["empty_question"] += 1
            continue
        if not (min_question_chars <= len(question) <= max_question_chars):
            drops["bad_question_len"] += 1
            continue

        wa_uids = _extract_with_answer_uids(row)
        if not wa_uids:
            drops["no_with_answer_paragraph"] += 1
            continue

        # Берём первый параграф с ответом, у которого подходящая длина.
        chosen_text: str | None = None
        chosen_uid: int | None = None
        for puid in wa_uids:
            txt = paragraphs_by_uid.get(puid)
            if txt is None:
                continue
            tlen = len(txt)
            if min_doc_chars <= tlen <= max_doc_chars:
                chosen_text = txt
                chosen_uid = puid
                break
        if chosen_text is None:
            # Откатимся на любой по длине — главное, чтобы был
            for puid in wa_uids:
                txt = paragraphs_by_uid.get(puid)
                if txt is not None:
                    chosen_text = txt
                    chosen_uid = puid
                    break

        if chosen_text is None:
            drops["paragraph_uid_not_in_corpus"] += 1
            continue
        if len(chosen_text) < min_doc_chars:
            drops["paragraph_too_short"] += 1
            continue
        if len(chosen_text) > max_doc_chars:
            # Усечём по последней точке в первых max_doc_chars символах.
            cut = chosen_text[:max_doc_chars]
            dot = cut.rfind(". ")
            if dot > min_doc_chars:
                chosen_text = cut[: dot + 1]
            else:
                chosen_text = cut

        pairs.append({
            "qid": f"rubq_{uid}",
            "query": question,
            "d_plus": chosen_text,
            "meta": {
                "rubq_uid": uid,
                "paragraph_uid": chosen_uid,
                "answer_text": answer,
                "tags": tags,
                "rubq_version": row.get("RuBQ_version"),
                "source_split": row.get("_source_split"),
                "n_with_answer_candidates": len(wa_uids),
            },
        })

    return pairs, drops


# ──────────────────────────────────────────────────────────────────────
# Статистика
# ──────────────────────────────────────────────────────────────────────

def _percentiles(xs: list[float], pcts=(10, 25, 50, 75, 90)) -> dict:
    if not xs:
        return {f"p{p}": 0 for p in pcts}
    xs_sorted = sorted(xs)
    out = {}
    for p in pcts:
        # nearest-rank, без интерполяции — проще и достаточно
        k = max(0, min(len(xs_sorted) - 1, int(round(p / 100 * (len(xs_sorted) - 1)))))
        out[f"p{p}"] = xs_sorted[k]
    return out


def compute_stats(pairs: list[dict], drops: Counter) -> dict:
    q_lens = [len(p["query"]) for p in pairs]
    d_lens = [len(p["d_plus"]) for p in pairs]
    n_words_q = [len(p["query"].split()) for p in pairs]
    n_words_d = [len(p["d_plus"].split()) for p in pairs]

    tag_counter: Counter = Counter()
    for p in pairs:
        tags = p["meta"].get("tags") or []
        if not tags:
            tag_counter["<no-tag>"] += 1
        for t in tags:
            tag_counter[t] += 1

    split_counter = Counter(p["meta"].get("source_split", "?") for p in pairs)

    return {
        "n_pairs": len(pairs),
        "drop_reasons": dict(drops),
        "query_chars": {
            "mean": round(mean(q_lens), 1) if q_lens else 0,
            "median": median(q_lens) if q_lens else 0,
            "stdev": round(stdev(q_lens), 1) if len(q_lens) > 1 else 0,
            **_percentiles(q_lens),
            "min": min(q_lens) if q_lens else 0,
            "max": max(q_lens) if q_lens else 0,
        },
        "d_plus_chars": {
            "mean": round(mean(d_lens), 1) if d_lens else 0,
            "median": median(d_lens) if d_lens else 0,
            "stdev": round(stdev(d_lens), 1) if len(d_lens) > 1 else 0,
            **_percentiles(d_lens),
            "min": min(d_lens) if d_lens else 0,
            "max": max(d_lens) if d_lens else 0,
        },
        "query_words": {
            "mean": round(mean(n_words_q), 1) if n_words_q else 0,
            "median": median(n_words_q) if n_words_q else 0,
        },
        "d_plus_words": {
            "mean": round(mean(n_words_d), 1) if n_words_d else 0,
            "median": median(n_words_d) if n_words_d else 0,
        },
        "tags": dict(tag_counter),
        "splits": dict(split_counter),
    }


# ──────────────────────────────────────────────────────────────────────
# Сэмплирование dev-set
# ──────────────────────────────────────────────────────────────────────

def sample_dev_set(
    pairs: list[dict], k: int, seed: int = 42
) -> list[dict]:
    """
    Берём детерминированный сэмпл k штук, по возможности стратифицированный по тегам.
    Если тегов меньше k, добивает рандомно.
    """
    rng = random.Random(seed)
    by_tag: dict[str, list[dict]] = {}
    for p in pairs:
        tags = p["meta"].get("tags") or ["<no-tag>"]
        primary = tags[0]  # достаточно, теги в RuBQ обычно одиночные
        by_tag.setdefault(primary, []).append(p)

    # Поровну на тег, остаток равномерно распределяем по тегам
    tag_keys = list(by_tag.keys())
    base = k // len(tag_keys)
    rem = k - base * len(tag_keys)

    chosen: list[dict] = []
    for i, tk in enumerate(tag_keys):
        bucket = list(by_tag[tk])
        rng.shuffle(bucket)
        n_take = base + (1 if i < rem else 0)
        chosen.extend(bucket[: min(n_take, len(bucket))])

    # Если из-за маленьких бакетов не добрали — добиваем из оставшихся
    if len(chosen) < k:
        chosen_set = {p["qid"] for p in chosen}
        rest = [p for p in pairs if p["qid"] not in chosen_set]
        rng.shuffle(rest)
        chosen.extend(rest[: k - len(chosen)])

    rng.shuffle(chosen)
    return chosen[:k]


# ──────────────────────────────────────────────────────────────────────
# I/O
# ──────────────────────────────────────────────────────────────────────

def save_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--splits",
        nargs="+",
        default=["dev", "test"],
        choices=["dev", "test"],
        help="какие сплиты RuBQ загружать (по умолчанию оба)",
    )
    ap.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "data" / "rubq"),
        help="куда писать rubq_full.jsonl, rubq_dev50.jsonl, stats.json",
    )
    ap.add_argument(
        "--dev-k", type=int, default=50,
        help="размер dev-сэмпла для Фазы 2",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--min-doc-chars", type=int, default=80,
        help="минимальная длина d⁺ в символах",
    )
    ap.add_argument(
        "--max-doc-chars", type=int, default=2500,
        help="максимальная длина d⁺ в символах (длиннее — усечём по точке)",
    )
    ap.add_argument(
        "--hf-cache",
        default=os.environ.get("HF_HOME"),
        help="HuggingFace cache dir; если не задан, берётся из HF_HOME",
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )

    q_records, p_index = load_datasets(args.splits, args.hf_cache)
    log.info("Всего вопросов из выбранных сплитов: %d", len(q_records))

    pairs, drops = build_pairs(
        q_records, p_index,
        min_doc_chars=args.min_doc_chars,
        max_doc_chars=args.max_doc_chars,
    )
    stats = compute_stats(pairs, drops)
    log.info("Валидных пар: %d", len(pairs))
    log.info("Отбрасывали по причинам: %s", dict(drops))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_jsonl(out_dir / "rubq_full.jsonl", pairs)
    log.info("Записано: %s", out_dir / "rubq_full.jsonl")

    if len(pairs) >= args.dev_k:
        dev = sample_dev_set(pairs, k=args.dev_k, seed=args.seed)
        save_jsonl(out_dir / f"rubq_dev{args.dev_k}.jsonl", dev)
        log.info(
            "Сэмпл dev (k=%d, seed=%d): %s",
            args.dev_k, args.seed, out_dir / f"rubq_dev{args.dev_k}.jsonl",
        )
    else:
        log.warning(
            "Валидных пар %d < dev_k=%d — пропускаем сэмплирование",
            len(pairs), args.dev_k,
        )

    (out_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    # Краткая сводка
    print("\n" + "=" * 60)
    print("PHASE 1 — RUBQ LOAD SUMMARY")
    print("=" * 60)
    print(f"Валидных пар:      {stats['n_pairs']}")
    print(f"Сплиты:            {stats['splits']}")
    print(f"Теги:              {stats['tags']}")
    print()
    print("Запрос (символов):")
    qs = stats["query_chars"]
    print(f"  mean={qs['mean']}  median={qs['median']}  p10={qs['p10']}  p90={qs['p90']}")
    print(f"  min={qs['min']}    max={qs['max']}")
    print("d⁺ (символов):")
    ds = stats["d_plus_chars"]
    print(f"  mean={ds['mean']}  median={ds['median']}  p10={ds['p10']}  p90={ds['p90']}")
    print(f"  min={ds['min']}    max={ds['max']}")
    print()
    print(f"Отброшено по причинам:")
    for k, v in stats["drop_reasons"].items():
        print(f"  {k}: {v}")
    print()
    print(f"Файлы в {out_dir}:")
    for f in sorted(out_dir.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
