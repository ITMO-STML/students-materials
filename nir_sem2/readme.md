QA Dataset Processing Toolkit

Описание
Набор инструментов для обработки датасетов вопросов и ответов (QA) различных типов:

1. Датасеты с **одним правильным ответом** на вопрос
2. Датасеты с **несколькими правильными ответами** на вопрос
3. Датасеты с **множественным выбором** (один верный из нескольких)
4. **Потоковая загрузка** больших датасетов
5. **Валидация и очистка** данных

Установка

pip install datasets huggingface-hub zstandard pandas clearml

Основные функции

1. Обработка датасетов с одним ответом
process_single_answer_dataset() - для датасетов с одним правильным ответом на вопрос.

Пример использования:

dataset, corpus = process_single_answer_dataset(

    dataset_name="squad",
    
    out_dir="./processed_squad",
    
    question_columns="question",
    
    answer_columns="answers")
2. Обработка датасетов с несколькими ответами

process_qa_dataset_with_multiple_answers() - когда на вопрос есть несколько верных ответов.

Пример:

dataset, corpus = process_qa_dataset_with_multiple_answers(

    dataset_name="hotpot_qa",
    
    out_dir="./processed_hotpot")
    
3. Обработка датасетов с множественным выбором

process_multi_answer_dataset() - для вопросов с вариантами ответов.

Пример:

dataset, corpus = process_multi_answer_dataset(
    dataset_path="my_qa_dataset",
    output_dir="./processed_multi_choice",
    
    question_field="question",
    
    answers_field="choices")
    

4. Потоковая загрузка данных

stream_zst_dataset() - для работы с большими сжатыми датасетами.


Пример:

dataset = Dataset.from_generator(
    stream_zst_dataset,
    
    gen_kwargs={
    
        "repo_id": "user/dataset",
        
        "filename": "data.json.zst"
        
    })

5. Валидация датасетов

validate_qa_datasets() - проверка целостности данных.

Пример:

issues = validate_qa_datasets(

    datasets_base_dir="./processed",
    
    dataset_names=["squad", "hotpot"])

Входные данные
Поддерживаются:

Датсеты из Hugging Face Hub

Локальные датасеты в формате Hugging Face Dataset

Сжатые файлы в формате .zst

Выходные данные

Структура после обработки:

text

output_dir/

├── corpus/          # Все ответы с уникальными ID

├── dataset/         # Вопросы с привязкой к ответам

│   ├── train

│   ├── val

│   └── test

└── cleaned/         # Очищенные версии (опционально)

Дополнительные возможности

Автоматическое определение train/val/test split

Логирование в ClearML

Обработка вложенных структур ответов

Генерация меток для оценки качества

Требования
Python 3.7+

Библиотеки:

datasets

huggingface-hub

zstandard

pandas

clearml (опционально)


