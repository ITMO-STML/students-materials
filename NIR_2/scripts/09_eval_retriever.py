#!/usr/bin/env python3
"""
Phase PoC: eval dense retriever на test200 vs corpus2096.

ЧТО ЭТО.
  Загружает saved model (от 08_train_retriever) или базовую RoSBERTa,
  кодирует corpus (2096 docs) + test queries (200), считает retrieval
  metrics: Recall@K, MRR@10, NDCG@10, Hits@10.

  Этот скрипт работает с ОДНОЙ моделью (одно condition или baseline).
  10_compare_ablations.py агрегирует результаты по всем conditions.

ВХОД.
  - data/train/test.jsonl    — 200 test queries с gold_qid.
  - data/train/corpus.jsonl  — 2096 docs (qid, text).
  - Saved model path или HuggingFace model name (default — fine-tuned).

ВЫХОД (outputs/eval/<condition>_seed<N>/).
  per_query.jsonl       — для каждого test query: top-K predicted, gold_rank.
  summary.json          — агрегированные metrics + config.
  report.md             — человекочитаемый отчёт.

МЕТРИКИ (стандарт retrieval evaluation).
  - Recall@K: доля queries у которых gold документ попал в top-K.
  - MRR@10:   mean reciprocal rank (1/rank если gold в top-10, иначе 0).
  - NDCG@10:  нормализованный gain. Для single-relevant gold это
              1/log2(rank+1) если gold в top-10, иначе 0.
  - Hits@10:  то же что Recall@10 (для совместимости с разной литературой).

ЗАПУСК.
  # Eval saved модель condition A:
  python scripts/09_eval_retriever.py --condition A --seed 42

  # Eval baseline (без fine-tune):
  python scripts/09_eval_retriever.py --baseline

  # Eval batch всех 5 conditions:
  python scripts/09_eval_retriever.py --conditions A B C D E --seed 42

  # Custom model path:
  python scripts/09_eval_retriever.py --model-path /path/to/saved/model --tag custom
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

log = logging.getLogger("eval")

CONDITION_TAGS = ["A", "B", "C", "D", "E"]


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
# Encoding
# ──────────────────────────────────────────────────────────────────────

def encode_texts(model, texts: list[str], batch_size: int = 64) -> "np.ndarray":
    """Кодирует список текстов, возвращает (N, dim) NumPy array (normalized)."""
    return model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )


# ──────────────────────────────────────────────────────────────────────
# Retrieval + metrics
# ──────────────────────────────────────────────────────────────────────

def retrieve_top_k(
    query_emb: "np.ndarray",
    corpus_emb: "np.ndarray",
    corpus_qids: list[str],
    top_k: int = 10,
) -> list[list[tuple[str, float]]]:
    """
    Для каждого query возвращает список [(qid, sim_score), ...] длины top_k.
    Работает батчами для экономии памяти.
    """
    import numpy as np

    n_queries = query_emb.shape[0]
    out: list[list[tuple[str, float]]] = []

    # Считаем similarities матрицей; для 200×2096 это 400K floats — OK в памяти
    sims = query_emb @ corpus_emb.T   # (n_q, n_c)

    # Argpartition для top-K быстрее чем full argsort
    top_idx = np.argpartition(-sims, top_k, axis=1)[:, :top_k]
    for i in range(n_queries):
        row = sims[i]
        idxs = top_idx[i]
        # Сортируем эти top-K чтобы получить правильный порядок
        idxs_sorted = idxs[np.argsort(-row[idxs])]
        out.append([(corpus_qids[j], float(row[j])) for j in idxs_sorted])
    return out


def compute_metrics(per_query: list[dict], top_k: int = 10) -> dict:
    """
    Стандартные retrieval metrics.

    Для каждого query gold_rank ∈ {1, ..., top_k} or None.
      - Recall@K = доля queries с gold_rank ≤ K
      - MRR@K   = mean(1/gold_rank если ≤ K else 0)
      - NDCG@K  = mean(1/log2(gold_rank+1) если ≤ K else 0)
                  (для single-relevant)
    """
    n = len(per_query)
    if n == 0:
        return {"n_queries": 0}

    ks = sorted(k for k in (1, 3, 5, 10) if k <= top_k)
    recall_hits = {k: 0 for k in ks}
    mrr_sum = 0.0
    ndcg_sum = 0.0
    ranks_when_hit = []

    for q in per_query:
        rank = q.get("gold_rank")
        if rank is None or rank > top_k:
            continue
        for k in ks:
            if rank <= k:
                recall_hits[k] += 1
        mrr_sum += 1.0 / rank
        ndcg_sum += 1.0 / math.log2(rank + 1)
        ranks_when_hit.append(rank)

    metrics = {"n_queries": n, "top_k": top_k}
    for k in ks:
        metrics[f"recall@{k}"] = round(recall_hits[k] / n, 4)
    metrics[f"mrr@{top_k}"]  = round(mrr_sum / n, 4)
    metrics[f"ndcg@{top_k}"] = round(ndcg_sum / n, 4)
    metrics[f"hits@{top_k}"] = round(recall_hits[max(ks)] / n, 4)
    if ranks_when_hit:
        metrics["mean_rank_when_hit"] = round(sum(ranks_when_hit) / len(ranks_when_hit), 2)
        metrics["median_rank_when_hit"] = sorted(ranks_when_hit)[len(ranks_when_hit) // 2]

    return metrics


# ──────────────────────────────────────────────────────────────────────
# Eval one model
# ──────────────────────────────────────────────────────────────────────

def eval_one_model(
    *,
    model_path: str,
    test_queries: list[dict],
    corpus: list[dict],
    output_dir: Path,
    top_k: int,
    batch_size: int,
    tag: str,
) -> dict:
    """Кодирует corpus + queries, ищет top-K, считает metrics, пишет outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Загрузка модели
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "sentence-transformers не установлен: pip install sentence-transformers"
        ) from e

    log.info("[%s] loading model: %s", tag, model_path)
    t0 = time.time()
    model = SentenceTransformer(model_path)
    log.info("[%s] model loaded in %.1fs", tag, time.time() - t0)

    # 2. Кодируем corpus
    log.info("[%s] encoding corpus (%d docs)...", tag, len(corpus))
    corpus_texts = [c["text"] for c in corpus]
    corpus_qids = [c["qid"] for c in corpus]
    t0 = time.time()
    corpus_emb = encode_texts(model, corpus_texts, batch_size=batch_size)
    t_corpus = time.time() - t0
    log.info("[%s] corpus encoded in %.1fs (dim=%d)",
             tag, t_corpus, corpus_emb.shape[1])

    # 3. Кодируем queries
    log.info("[%s] encoding %d test queries...", tag, len(test_queries))
    query_texts = [q["query"] for q in test_queries]
    t0 = time.time()
    query_emb = encode_texts(model, query_texts, batch_size=batch_size)
    t_query = time.time() - t0
    log.info("[%s] queries encoded in %.1fs", tag, t_query)

    # 4. Retrieval
    log.info("[%s] retrieving top-%d...", tag, top_k)
    t0 = time.time()
    top_k_results = retrieve_top_k(query_emb, corpus_emb, corpus_qids, top_k=top_k)
    t_retrieve = time.time() - t0
    log.info("[%s] retrieved in %.2fs", tag, t_retrieve)

    # 5. Per-query records
    per_query = []
    for q, hits in zip(test_queries, top_k_results):
        gold_qid = q["gold_qid"]
        predicted_qids = [h[0] for h in hits]
        try:
            gold_rank = predicted_qids.index(gold_qid) + 1
        except ValueError:
            gold_rank = None
        per_query.append({
            "qid": q["qid"],
            "query": q["query"],
            "gold_qid": gold_qid,
            "gold_rank": gold_rank,
            "gold_in_top_k": gold_rank is not None,
            "top_k_predictions": [
                {"qid": qid, "score": round(score, 4)} for qid, score in hits
            ],
        })

    # 6. Metrics
    metrics = compute_metrics(per_query, top_k=top_k)

    # 7. Save outputs
    save_jsonl(output_dir / "per_query.jsonl", per_query)
    summary = {
        "tag": tag,
        "model_path": model_path,
        "config": {
            "top_k": top_k,
            "batch_size": batch_size,
        },
        "data": {
            "n_test_queries": len(test_queries),
            "n_corpus": len(corpus),
        },
        "metrics": metrics,
        "timing_seconds": {
            "corpus_encode": round(t_corpus, 2),
            "query_encode":  round(t_query, 2),
            "retrieve":      round(t_retrieve, 2),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


# ──────────────────────────────────────────────────────────────────────
# Report.md
# ──────────────────────────────────────────────────────────────────────

def write_report(summary: dict, output_dir: Path, per_query: list[dict]) -> None:
    tag = summary["tag"]
    m = summary["metrics"]
    n = m["n_queries"]

    lines = []
    lines.append(f"# Retrieval Eval — {tag}\n")
    lines.append(f"- Model: `{summary['model_path']}`")
    lines.append(f"- Test queries: **{n}**")
    lines.append(f"- Corpus size: **{summary['data']['n_corpus']}**")
    lines.append(f"- top_k: {m['top_k']}")
    lines.append("")

    lines.append("## Metrics\n")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    for key in sorted(m.keys()):
        if key in ("n_queries", "top_k"):
            continue
        v = m[key]
        if isinstance(v, float):
            lines.append(f"| {key} | {v:.4f} |")
        else:
            lines.append(f"| {key} | {v} |")
    lines.append("")

    # Timing
    lines.append("## Timing\n")
    t = summary["timing_seconds"]
    lines.append(f"- corpus encode: {t['corpus_encode']}s")
    lines.append(f"- query encode:  {t['query_encode']}s")
    lines.append(f"- retrieve:      {t['retrieve']}s")
    lines.append("")

    # Failure analysis
    misses = [q for q in per_query if q["gold_rank"] is None]
    lines.append("## Failure analysis\n")
    lines.append(f"- {len(misses)}/{n} queries не нашли gold в top-{m['top_k']}")
    if misses[:3]:
        lines.append("\n**Примеры miss queries:**\n")
        for q in misses[:3]:
            lines.append(f"- **{q['qid']}**: {q['query']}")
            lines.append(f"  - gold_qid: `{q['gold_qid']}` (rank не в top-{m['top_k']})")
            top_qid = q["top_k_predictions"][0]["qid"]
            top_score = q["top_k_predictions"][0]["score"]
            lines.append(f"  - top-1 predicted: `{top_qid}` (score={top_score})")
    lines.append("")

    # Hits with good rank
    hits = [q for q in per_query if q["gold_rank"] == 1]
    lines.append(f"## Perfect hits (rank=1): {len(hits)}/{n}\n")

    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def resolve_model_path(args, condition: str | None) -> tuple[str, str]:
    """
    Возвращает (model_path, tag).
    Приоритет: --model-path > --baseline > --condition.
    """
    if args.model_path:
        tag = args.tag or "custom"
        return args.model_path, tag
    if args.baseline:
        return args.base_model, "BASE"
    if condition:
        path = Path(args.training_base) / f"{condition}_seed{args.seed}" / "model"
        if not path.exists():
            raise FileNotFoundError(
                f"Saved model не найден: {path}\n"
                f"Сначала запусти 08_train_retriever.py --condition {condition}"
            )
        return str(path), condition
    raise ValueError("Нужен --model-path, --baseline, или --condition")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir",
                    default=str(PROJECT_ROOT / "data/train"))
    ap.add_argument("--training-base",
                    default=str(PROJECT_ROOT / "outputs/training"),
                    help="где лежат saved модели от 08")
    ap.add_argument("--output-base",
                    default=str(PROJECT_ROOT / "outputs/eval"))
    ap.add_argument("--base-model",
                    default="ai-forever/ru-en-RoSBERTa")
    ap.add_argument("--condition", choices=CONDITION_TAGS)
    ap.add_argument("--conditions", nargs="+", choices=CONDITION_TAGS,
                    help="batch eval всех указанных")
    ap.add_argument("--baseline", action="store_true",
                    help="eval базовой модели (без fine-tune)")
    ap.add_argument("--model-path", default=None,
                    help="прямой путь к saved model или HF model name")
    ap.add_argument("--tag", default=None,
                    help="имя тега для output dir (при --model-path)")
    ap.add_argument("--seed", type=int, default=42,
                    help="seed, использованный в 08 (для resolve пути)")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Решаем какие модели обрабатывать
    targets: list[tuple[str, str]] = []  # (model_path, tag)
    if args.conditions:
        for cond in args.conditions:
            path, tag = resolve_model_path(args, cond)
            targets.append((path, f"{tag}_seed{args.seed}"))
    elif args.model_path or args.baseline or args.condition:
        path, tag = resolve_model_path(args, args.condition)
        suffix = f"_seed{args.seed}" if args.condition else ""
        targets.append((path, f"{tag}{suffix}"))
    else:
        ap.error("Нужен один из: --condition, --conditions, --baseline, --model-path")

    # Загрузка тестовых данных (один раз)
    train_dir = Path(args.train_dir)
    test_path = train_dir / "test.jsonl"
    corpus_path = train_dir / "corpus.jsonl"
    if not test_path.exists() or not corpus_path.exists():
        log.error("Не найдены test.jsonl/corpus.jsonl в %s. Сначала 07_build_train_data.py", train_dir)
        sys.exit(1)

    test_queries = load_jsonl(test_path)
    corpus = load_jsonl(corpus_path)
    log.info("Loaded test=%d, corpus=%d", len(test_queries), len(corpus))

    # Eval каждой модели
    all_summaries = []
    output_base = Path(args.output_base)
    for model_path, tag in targets:
        out_dir = output_base / tag
        log.info("=" * 60)
        log.info("Evaluating %s", tag)
        log.info("=" * 60)
        try:
            summary = eval_one_model(
                model_path=model_path,
                test_queries=test_queries,
                corpus=corpus,
                output_dir=out_dir,
                top_k=args.top_k,
                batch_size=args.batch_size,
                tag=tag,
            )
            per_query = load_jsonl(out_dir / "per_query.jsonl")
            write_report(summary, out_dir, per_query)
            all_summaries.append(summary)
        except Exception as e:
            log.exception("[%s] eval failed: %s", tag, e)

    # ──────────────────────────────────────────────
    # Финальный stdout
    # ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("EVAL SUMMARY")
    print("=" * 70)
    print(f"{'tag':<20} {'recall@1':>10} {'recall@5':>10} "
          f"{'recall@10':>11} {'mrr@10':>9} {'ndcg@10':>9}")
    for s in all_summaries:
        m = s["metrics"]
        print(f"{s['tag']:<20} "
              f"{m.get('recall@1', 0):>10.4f} "
              f"{m.get('recall@5', 0):>10.4f} "
              f"{m.get('recall@10', 0):>11.4f} "
              f"{m.get('mrr@10', 0):>9.4f} "
              f"{m.get('ndcg@10', 0):>9.4f}")
    print()
    print("Files:")
    for s in all_summaries:
        out_dir = output_base / s["tag"]
        print(f"  {out_dir}/")
        print(f"    summary.json, report.md, per_query.jsonl")
    print()
    print("→ Next: 10_compare_ablations.py для финальной таблицы")


if __name__ == "__main__":
    main()
