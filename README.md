# Бенчмарки генерации кода (NIR)

Репозиторий разбит на три логических блока:

| Папка | Назначение | Точка входа |
|-------|------------|-------------|
| **dataset_downloader/** | Скачивание и пред‑обработка датасетов (HumanEval, MBPP, CoNaLa и др.) в `.jsonl` | `python -m dataset_downloader.dataset_downloader` |
| **experiment/** | Базовый запуск *vanilla‑prompt* | `experiment/run_autoprompt_experiment.ipynb` |
| **new_exp/** | Пользовательские эксперименты (AutoPrompt, soft‑prompt, RAG‑prompt …) | `new_exp/run_autoprompt_full.ipynb` |

> **TL;DR** — создайте виртуальное окружение, скачайте данные, исправьте пути в YAML‑файлах и запустите ноутбуки. GPU (≥8ГБ VRAM) желателен, но не обязателен.

---

## 1 Установка окружения

```bash
git clone <URL-репозитория>
cd <repo>
python -m venv .venv
source .venv/bin/activate 
pip install -U pip
pip install -r requirements.txt  
```

---

## 2 Загрузка и подготовка датасетов

1. **Укажите каталог для данных.**

   В `dataset_downloader/dataser_downloader.py` по умолчанию прописан путь:

   ```python
   BASE = Path("G:/ITMO/NIR/DATASETS")
   ```

   Измените константу **или** выставьте переменную окружения:

   ```bash
   export NIR_DATASETS=/absolute/path/to/datasets
   ```

2. **Скачайте и пред‑обработайте данные** (≈5‑10мин):

   ```bash
   python -m dataset_downloader.dataset_downloader
   # Создаст:
   #   $BASE/processed/*.jsonl — готовые выборки
   #   $BASE/hf_cache/ 
   ```

---

## 3 Baseline‑эксперимент

1. Откройте `experiment/baseline_prompt.yaml` и поправьте поле `root:`:

   ```yaml
   root: "/absolute/path/to/datasets/processed"
   ```

2. Запустите ноутбук:

   ```bash
   jupyter lab
   # Файл: experiment/run_autoprompt_experiment.ipynb → «Run ▶︎ All»
   ```

3. Результаты сохранятся в `experiment/results/`
   (предсказания, метрики CSV, графики PNG).

---

## 4 Собственные эксперименты

1. Отредактируйте `new_exp/big_run.yaml` под свои нужды:

   ```yaml
   batch: 1              # градиент‑аккумуляция
   n_iters: 32           # шаги оптимизации AutoPrompt
   sample_size: null     # null → весь датасет
   ```

2. Запустите:

   ```bash
   jupyter lab
   # Файл: new_exp/run_autoprompt_full.ipynb → «Run ▶︎ All»
   ```

3. Артефакты появятся в `new_exp/results/`.

---

## 5 Визуализация результатов

```bash
python new_exp/plot_autoprompt_results.py
```

Скрипт соберёт все `*.csv`/`*.jsonl` из `**/results/` и положит PNG‑графики рядом.

---

## 6 Структура проекта

```
.
├── dataset_downloader/
│   ├── data/ …
│   └── dataser_downloader.py
├── experiment/
│   ├── baseline_prompt.yaml
│   ├── run_autoprompt_experiment.ipynb
│   └── results/
├── new_exp/
│   ├── big_run.yaml
│   ├── run_autoprompt_full.ipynb
│   ├── plot_autoprompt_results.py
│   └── results/
└── requirements.txt
```

---

## 7 Типичные проблемы

| Ошибка | Причина | Решение |
|--------|---------|---------|
| `FileNotFoundError: *.jsonl` | Неверный путь `root:` в YAML | Исправить абсолютный путь |
| `CUDA out of memory` | Модель/батч не помещается в память GPU | Уменьшить `batch`, `beam_size` или перейти на CPU |
| `ModuleNotFoundError` | venv не активирован или пакеты не установлены | `source .venv/bin/activate && pip install -r requirements.txt` |

---
