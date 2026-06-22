# Многомодальный анализ тональности и распознавание эмоций на корпусе CMU-MOSEI

Курсовая работа: построение многозадачной нейросетевой архитектуры для одновременной классификации тональности (3 класса) и распознавания эмоций (6 multilabel-меток) на корпусе CMU-MOSEI с использованием двух модальностей: текстовой (GloVe-эмбеддинги, TF-IDF) и акустической (COVAREP).

## Структура репозитория

```
.
├── src/                    # Исходный код модулей
│   ├── data_loader.py      # Загрузка корпуса, интеграция меток (Hungarian algorithm)
│   ├── features.py         # Извлечение признаков, стандартизация
│   ├── models.py           # Базовая FFN-архитектура
│   ├── advanced_models.py  # BiLSTM, CNN1D, Transformer, fusion-модели
│   ├── train.py            # Циклы обучения, multitask-loss с маскированием
│   └── metrics.py          # Метрики классификации (macro-F1, Acc-2, ROC-AUC)
├── notebooks/              # Jupyter-ноутбуки
│   ├── 01_eda.ipynb        # Эксплораторный анализ корпуса
│   ├── 02_baseline.ipynb   # Базовые модели прямого распространения
│   ├── 03_advanced_models.ipynb  # BiLSTM/CNN1D/Transformer + grid search
│   └── 04_fusion_multitask.ipynb # Late/Early/Cross-modal объединение
├── experiments/            # Результаты экспериментов в .csv
├── figures/                # Графики EDA и схемы архитектур
├── report/                 # Финальный отчёт по ГОСТ 7.32
├── requirements.txt        # Зависимости
└── README.md
```

## Данные

Корпус CMU-MOSEI не включён в репозиторий из-за размера и условий распространения. Для воспроизведения экспериментов скачайте признаки самостоятельно:

1. `mosei_senti_data.pkl` (текст GloVe + аудио COVAREP + видео FACET + sentiment) — из репозитория [MultiBench](https://github.com/pliang279/MultiBench).
2. `mosei.hdf5` (эмоциональные метки в формате CMU-MultimodalSDK) — из [CMU-MultiComp Lab](http://immortal.multicomp.cs.cmu.edu/raw_datasets/processed_data/cmu-mosei/).

Поместите оба файла в `data/raw/`. Кэшированные промежуточные результаты в `data/processed/` создаются автоматически при первом запуске ноутбука `01_eda.ipynb`.

## Запуск

```bash
pip install -r requirements.txt
jupyter notebook
```

Ноутбуки запускаются последовательно: 01 → 02 → 03 → 04.

## Основные результаты

| Модель | Модальности | Sentiment F1 | Emotion F1 | Mean AUC |
|---|---|---|---|---|
| Лучшая базовая (GloVe + FFN) | текст | 0,589 | 0,409 | 0,685 |
| Transformer (настроенный) | текст | **0,602** | 0,413 | 0,686 |
| Transformer | аудио | 0,444 | 0,362 | 0,628 |
| Позднее объединение | текст + аудио | 0,577 | **0,424** | **0,705** |
| Раннее объединение | текст + аудио | 0,597 | 0,418 | 0,702 |
| Кросс-модальное (внимание) | текст + аудио | 0,600 | 0,420 | 0,704 |

Поставленная цель (F1 ≥ 0,60 для тональности и ≥ 0,42 для эмоций) достигнута.

## Технологии

PyTorch, scikit-learn, scipy, NumPy, pandas, matplotlib, h5py.

## Автор

Попов Александр, ИТМО.