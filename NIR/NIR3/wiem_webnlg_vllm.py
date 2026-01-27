#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===================== КОМАНДЫ ДЛЯ ЗАПУСКА =====================

1) (РЕКОМЕНДУЕТСЯ) создать и активировать виртуальное окружение
--------------------------------------------------------------
python3 -m venv venv
source venv/bin/activate

2) Установить зависимости Python
--------------------------------
pip install --upgrade pip

pip install \
  datasets \
  pandas \
  openai \
  tenacity \
  tqdm \
  vllm \
  transformers \
  accelerate

3) Запустить LLM-сервер vLLM с моделью Qwen2.5-32B-Instruct
-----------------------------------------------------------
ВАЖНО:
- требуется GPU (A100/H100 или эквивалент)
- модель будет автоматически скачана с Hugging Face

vllm serve Qwen/Qwen2.5-32B-Instruct \
  --dtype auto \
  --max-model-len 8192

После этого сервер будет доступен по адресу:
http://localhost:8000/v1

4) В ДРУГОМ терминале запустить этот скрипт
------------------------------------------
Пример запуска на подвыборке WebNLG:

python wiem_webnlg_vllm.py \
  --split "train[:50]" \
  --out wiem_results.csv

==============================================================

Описание:
- Датасет WebNLG загружается внутри кода через Hugging Face Datasets
- Модель Qwen2.5-32B-Instruct загружается и обслуживается vLLM-сервером
- Скрипт реализует WIEM-подобную оценку полноты графов знаний
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from datasets import load_dataset
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm


@dataclass
class Config:
    """
    Центральная конфигурация эксперимента.
    Все ключевые параметры вынесены сюда.
    """
    dataset_name: str = "webnlg-challenge/web_nlg"
    split: str = "train[:50]"
    lang: Optional[str] = None

    base_url: str = "http://localhost:8000/v1"
    model_id: str = "Qwen/Qwen2.5-32B-Instruct"

    temperature: float = 0.0
    top_p: float = 1.0
    max_new_tokens: int = 700

    max_entities: int = 30
    max_triples_in_prompt: int = 25
    max_missing_return: int = 25

    cache_db: str = "wiem_cache.sqlite"
    out_csv: str = "wiem_results.csv"


def clean_node(x: str) -> str:
    """
    Нормализация имени сущности:
    - удаляем кавычки
    - обрезаем пробелы
    """
    x = str(x).strip()
    if len(x) >= 2 and x[0] == x[-1] and x[0] in "\"'":
        x = x[1:-1].strip()
    return x


def normalize_predicate(p: str) -> str:
    """
    Приводим предикат к устойчивому виду
    для сравнения тройек.
    """
    p = str(p).strip().lower()
    p = re.sub(r"\\s+", "_", p)
    return p


def normalize_triple(t: Tuple[str, str, str]) -> Tuple[str, str, str]:
    """
    Полная нормализация RDF-тройки.
    """
    s, p, o = t
    return (clean_node(s), normalize_predicate(p), clean_node(o))


def parse_triples(obj: Any) -> List[Tuple[str, str, str]]:
    """
    Универсальный парсер RDF-троек из WebNLG,
    так как формат может отличаться между версиями датасета.
    """
    triples = []

    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, str):
                parts = re.split(r"\\s*\\|\\s*", item)
                if len(parts) == 3:
                    triples.append((parts[0], parts[1], parts[2]))
            elif isinstance(item, dict):
                s = item.get("subject")
                p = item.get("predicate")
                o = item.get("object")
                if s and p and o:
                    triples.append((s, p, o))
            elif isinstance(item, (list, tuple)) and len(item) == 3:
                triples.append(tuple(item))

    elif isinstance(obj, dict):
        for key in ["modifiedtripleset", "originaltripleset", "triples"]:
            if key in obj:
                triples.extend(parse_triples(obj[key]))

    return triples


def extract_texts(example: Dict[str, Any]) -> List[str]:
    """
    Извлекает текстовые референсы для примера.
    """
    texts = []

    for key in ["text", "target"]:
        if isinstance(example.get(key), str):
            texts.append(example[key])

    entry = example.get("entry")
    if isinstance(entry, dict):
        lexs = entry.get("lexs")
        if isinstance(lexs, list):
            for l in lexs:
                if isinstance(l, dict) and "lex" in l:
                    texts.append(l["lex"])

    return [t for t in texts if t.strip()]


def extract_triples_and_meta(example: Dict[str, Any]):
    """
    Извлекает RDF-тройки и метаинформацию.
    """
    triples = []
    meta = {}

    if "entry" in example:
        entry = example["entry"]
        triples = parse_triples(entry)
        for k in ["category", "size", "shape", "shape_type"]:
            if k in entry:
                meta[k] = entry[k]
    else:
        triples = parse_triples(example)

    return triples, meta


def extract_entities(triples: List[Tuple[str, str, str]]) -> List[str]:
    """
    Формирует список уникальных сущностей графа.
    """
    entities = []
    seen = set()
    for s, _, o in triples:
        for e in (clean_node(s), clean_node(o)):
            if e and e not in seen:
                seen.add(e)
                entities.append(e)
    return entities


class Cache:
    """
    SQLite-кэш для результатов LLM,
    чтобы избежать повторных запросов.
    """

    def __init__(self, path: str):
        self.conn = sqlite3.connect(path)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT)"
        )

    def get(self, key: str) -> Optional[str]:
        cur = self.conn.execute("SELECT value FROM cache WHERE key = ?", (key,))
        row = cur.fetchone()
        return row[0] if row else None

    def set(self, key: str, value: str):
        self.conn.execute(
            "INSERT OR REPLACE INTO cache VALUES (?, ?)", (key, value)
        )
        self.conn.commit()


def make_cache_key(model: str, text: str, triples, entities) -> str:
    """
    Формирует ключ для кэша на основе входных данных.
    """
    payload = json.dumps(
        {"model": model, "text": text, "triples": triples, "entities": entities},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_prompt(text, entities, triples, cfg: Config):
    """
    Формирует промпт для LLM.
    """
    triples = triples[: cfg.max_triples_in_prompt]
    entities = entities[: cfg.max_entities]

    triples_str = "\\n".join(f"{s} | {p} | {o}" for s, p, o in triples)
    entities_str = "\\n".join(f"- {e}" for e in entities)

    return [
        {
            "role": "system",
            "content": (
                "Ты система извлечения отношений. "
                "Возвращай только корректный JSON."
            ),
        },
        {
            "role": "user",
            "content": f"""
ТЕКСТ:
{text}

СУЩНОСТИ:
{entities_str}

УЖЕ ИЗВЛЕЧЁННЫЕ TRIPLES:
{triples_str}

ЗАДАНИЕ:
Найди дополнительные отношения между сущностями,
которые явно поддерживаются текстом,
но отсутствуют в списке triples.

Верни JSON строго в формате:
{{
  "predicted_missing": [
    {{
      "s": "...",
      "p": "...",
      "o": "...",
      "confidence": 0.0,
      "evidence": "короткий фрагмент текста"
    }}
  ]
}}

Верни не более {cfg.max_missing_return} отношений.
Верни ТОЛЬКО JSON.
""",
        },
    ]


@retry(stop=stop_after_attempt(3), wait=wait_exponential())
def llm_predict_missing(client, cfg, text, entities, triples):
    """
    Вызов LLM для поиска пропущенных отношений.
    """
    response = client.chat.completions.create(
        model=cfg.model_id,
        messages=build_prompt(text, entities, triples, cfg),
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        max_tokens=cfg.max_new_tokens,
    )

    content = response.choices[0].message.content
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\\{.*\\}", content, re.S)
        if match:
            return json.loads(match.group(0))
        raise


def compute_wiem_completeness(
    gold_triples: List[Tuple[str, str, str]],
    predicted: List[Dict[str, Any]],
):
    """
    WIEM-подобная оценка полноты графа.
    """
    gold_set = set(normalize_triple(t) for t in gold_triples)

    total_mass = 0.0
    missing_mass = 0.0
    missing = []

    for t in predicted:
        s, p, o = t["s"], t["p"], t["o"]
        conf = float(t.get("confidence", 0.0))
        total_mass += conf

        if normalize_triple((s, p, o)) not in gold_set:
            missing_mass += conf
            missing.append((s, p, o, conf))

    completeness = 1.0 if total_mass == 0 else 1.0 - missing_mass / total_mass
    completeness = max(0.0, min(1.0, completeness))

    return completeness, missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train[:50]")
    parser.add_argument("--out", default="wiem_results.csv")
    args = parser.parse_args()

    cfg = Config(split=args.split, out_csv=args.out)

    client = OpenAI(base_url=cfg.base_url, api_key="EMPTY")
    cache = Cache(cfg.cache_db)

    dataset = load_dataset(cfg.dataset_name, split=cfg.split)

    rows = []

    for i, example in enumerate(tqdm(dataset)):
        triples, meta = extract_triples_and_meta(example)
        texts = extract_texts(example)

        if not triples or not texts:
            continue

        text = texts[0]
        entities = extract_entities(triples)

        cache_key = make_cache_key(cfg.model_id, text, triples, entities)
        cached = cache.get(cache_key)

        if cached:
            pred = json.loads(cached)
        else:
            pred = llm_predict_missing(client, cfg, text, entities, triples)
            cache.set(cache_key, json.dumps(pred, ensure_ascii=False))

        predicted = pred.get("predicted_missing", [])
        completeness, missing = compute_wiem_completeness(triples, predicted)

        rows.append(
            {
                "idx": i,
                "category": meta.get("category"),
                "n_triples": len(triples),
                "n_entities": len(entities),
                "completeness": completeness,
                "n_missing": len(missing),
                "top_missing": "; ".join(
                    f"{s}|{p}|{o}({c:.2f})" for s, p, o, c in missing[:5]
                ),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(cfg.out_csv, index=False)
    print(f"Результаты сохранены в {cfg.out_csv}")


if __name__ == "__main__":
    main()
