# Thesis Project

Проект для экспериментов с фонемной транскрипцией речи на нескольких подходах:

- `Baseline` модель
- `CTC` модель
- `CTC + Attention`
- отдельные эксперименты с `phonetic features`

## Структура проекта

В корне лежат общие модули, которые используются из ноутбуков и приложений:

- [app.py](/Users/ananeva/Downloads/Thesis_project_final/app.py) — текущее Gradio-приложение
- [project_paths.py](/Users/ananeva/Downloads/Thesis_project_final/project_paths.py) — единые пути к данным, моделям и артефактам
- [DatasetsModels.py](/Users/ananeva/Downloads/Thesis_project_final/DatasetsModels.py) — модели и датасеты
- [GetData.py](/Users/ananeva/Downloads/Thesis_project_final/GetData.py) — загрузка данных
- [Decoding_tools.py](/Users/ananeva/Downloads/Thesis_project_final/Decoding_tools.py) — декодирование и работа с границами
- [evaluation_tools.py](/Users/ananeva/Downloads/Thesis_project_final/evaluation_tools.py) — метрики и оценка
- [ToIPA.py](/Users/ananeva/Downloads/Thesis_project_final/ToIPA.py) — преобразование в IPA
- [SaveTools.py](/Users/ananeva/Downloads/Thesis_project_final/SaveTools.py) — сохранение результатов
- [VizulalisationTools.py](/Users/ananeva/Downloads/Thesis_project_final/VizulalisationTools.py) — визуализация
- [get_embeddings.py](/Users/ananeva/Downloads/Thesis_project_final/get_embeddings.py) — извлечение эмбеддингов

Папки проекта:

- [Baseline_Training](/Users/ananeva/Downloads/Thesis_project_final/Baseline_Training) — обучение, инференс и оценка baseline-модели
- [CTC_models](/Users/ananeva/Downloads/Thesis_project_final/CTC_models) — обучение и инференс CTC и CTC+Attention
- [PhoneticFeatures_Training](/Users/ananeva/Downloads/Thesis_project_final/PhoneticFeatures_Training) — эксперименты с фонетическими признаками
- [models](/Users/ananeva/Downloads/Thesis_project_final/models) — сохраненные рабочие модели
- [data](/Users/ananeva/Downloads/Thesis_project_final/data) — примеры, таблицы, логи и прочие артефакты
- [content_manager](/Users/ananeva/Downloads/Thesis_project_final/content_manager) — код для извлечения speech/content embeddings

## Данные и модели

### Данные

- [data/examples](/Users/ananeva/Downloads/Thesis_project_final/data/examples) — примеры `wav`, `seg`, `txt`, `npy`
- [data/outputs/tables](/Users/ananeva/Downloads/Thesis_project_final/data/outputs/tables) — сохраненные таблицы и результаты
- [data/outputs/logs](/Users/ananeva/Downloads/Thesis_project_final/data/outputs/logs) — логи обучения и экспериментов

### Модели

- [models/baseline](/Users/ananeva/Downloads/Thesis_project_final/models/baseline)
- [models/ctc](/Users/ananeva/Downloads/Thesis_project_final/models/ctc)
- [models/phonetic_features](/Users/ananeva/Downloads/Thesis_project_final/models/phonetic_features)

Пути по умолчанию задаются в [project_paths.py](/Users/ananeva/Downloads/Thesis_project_final/project_paths.py).

## Основные ноутбуки

### Baseline

- [Baseline_Training/main.ipynb](/Users/ananeva/Downloads/Thesis_project_final/Baseline_Training/main.ipynb) — основное обучение baseline
- [Baseline_Training/inference.ipynb](/Users/ananeva/Downloads/Thesis_project_final/Baseline_Training/inference.ipynb) — инференс baseline
- [Baseline_Training/evaluation.ipynb](/Users/ananeva/Downloads/Thesis_project_final/Baseline_Training/evaluation.ipynb) — оценка baseline
- [Baseline_Training/Extract Embeddings.ipynb](</Users/ananeva/Downloads/Thesis_project_final/Baseline_Training/Extract Embeddings.ipynb>) — извлечение эмбеддингов

### CTC и Attention

- [CTC_models/CTC_Training.ipynb](/Users/ananeva/Downloads/Thesis_project_final/CTC_models/CTC_Training.ipynb) — обучение CTC
- [CTC_models/CTC_Training_tacotron.ipynb](/Users/ananeva/Downloads/Thesis_project_final/CTC_models/CTC_Training_tacotron.ipynb) — обучение attention-модели
- [CTC_models/Attention_Inference.ipynb](/Users/ananeva/Downloads/Thesis_project_final/CTC_models/Attention_Inference.ipynb) — инференс attention-модели
- [CTC_models/evaluation_attention.ipynb](/Users/ananeva/Downloads/Thesis_project_final/CTC_models/evaluation_attention.ipynb) — оценка attention-модели

### Phonetic Features

- [PhoneticFeatures_Training/Phonetic_features_predictions_fixed.ipynb](/Users/ananeva/Downloads/Thesis_project_final/PhoneticFeatures_Training/Phonetic_features_predictions_fixed.ipynb)
- [PhoneticFeatures_Training/Phonetic_features_predictions_fixed_binary.ipynb](/Users/ananeva/Downloads/Thesis_project_final/PhoneticFeatures_Training/Phonetic_features_predictions_fixed_binary.ipynb)
- [PhoneticFeatures_Training/Phonetic_inference.ipynb](/Users/ananeva/Downloads/Thesis_project_final/PhoneticFeatures_Training/Phonetic_inference.ipynb)

## Запуск Gradio-приложения

Текущее приложение находится в [app.py](/Users/ananeva/Downloads/Thesis_project_final/app.py).

Что оно сейчас делает:

- загружает `wav`
- опционально загружает `seg_B2`
- запускает:
  - `Baseline`
  - `CTC`
  - `CTC + Attention`
- показывает:
  - транскрипцию
  - IPA
  - CER
  - Boundary Error
  - waveform + segmentation
  - alignment map только для `CTC + Attention`

Запуск:

```bash
python app.py
```

Если используется виртуальное окружение:

```bash
source venv/bin/activate
python app.py
```

На Windows:

```bash
python app.py
```

## Зависимости

Минимум нужны:

- `torch`
- `torchaudio`
- `gradio`
- `plotly`
- `numpy`
- `pandas`
- `panphon`
- `matplotlib`
- `seaborn`
- `python-Levenshtein`

Список пакетов см. в [requirements.txt](/Users/ananeva/Downloads/Thesis_project_final/requirements.txt).

## Замечания

- Проект ориентирован на запуск из корня репозитория.
- Для подпапок с кодом добавлены `__init__.py`, чтобы импорты были стабильнее.
- `content_manager/PhonHuBERT-main` — это отдельный большой блок стороннего/вендорного кода, его структура пока не была полностью перепакована под обычный Python package style.
- В `app.py` учтена совместимость со старыми версиями `Python 3.7` и `Gradio`.

## Быстрый сценарий работы

1. Подготовить окружение и установить зависимости.
2. Проверить, что модели лежат в [models](/Users/ananeva/Downloads/Thesis_project_final/models).
3. Проверить, что примерные данные лежат в [data/examples](/Users/ananeva/Downloads/Thesis_project_final/data/examples).
4. Запустить [app.py](/Users/ananeva/Downloads/Thesis_project_final/app.py) для демонстрации.
5. Использовать ноутбуки из `Baseline_Training` и `CTC_models` для обучения и анализа.
