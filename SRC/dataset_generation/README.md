# Генерация видео

## Создание новых текстовых описаний

В качестве основы используются наборы данных `MVBench` и `StreamingBench`.
Необходимо указать путь до набора данных, чтобы извлечь нужные темы.
Пример для `MVBench.tsv`:
```shell
python extract_video.py --dataset_path "MVBench.tsv" --output_path "cleaned_mvbench.tsv"
```

Затем можно генерировать новые описания для I2V с помощью `Qwen3-VL-8B`:
```shell
python recaptioning.py --file_path "cleaned_mvbench.tsv --type "MVBench"
```

## Удаление дубликатов

После генерации надо убедиться, что новые описания достаточно уникальны.
Делается это с помощью `LSH` и `MinHash` алгоритмов:

```shell
cd deduplication
python deduplicator.py --json_path "MVBench_cleaned.json" --save_path "MVBench_dedup.json"
```

Скрипт автоматически удалит примеры, у которых похожие описания для изображений, видео и изображений+видео.

## Переписывание

Использование модели `DeepSeek V3` для переписывания описания, а также генерации реальных и фейковых объектов:
```shell
python deepseek_rewrite.py --data_path "MVBench_dedup.json" --save_path "MVBench_rewrite.json" --url "url::/api/chat/completions" --key "sk-API-KEY"
```

## Фильтрация с помощью эмбедингов

`Qwen3-VL-Embedding` переводит реальные объекты в эмбеденги, а затем вычисляется косинусная близость для каждого фейкового объекта с каждым реальным:
```shell
python verifi_objects.py --embeder_path Qwen/Qwen3-VL-Embedding-8B --dataset_path "MVBench_rewrite.json" --save_path "MVBench_emb.json"
```

## Генерация изображений

Для генерации изображений используется модель `Z-Image` с негативной инструкцией в виде фейковых объектов:

```shell
python img_gen.py --model_path "Tongyi-MAI/Z-Image" --json_path "MVBench_emb.json" --save_folder "./generated_images/"
```

## I2V генерация

Склонируйте LightX2V репозиторий и активируйте коммит с хэшем `2f952497d4c1fe94dd4555fc7abdce9ef0a473d1`.
Скопируйте `init_vid_gen_venv.sh` в репозиторий.
Создайте конда-окружение.
```shell
git clone https://github.com/ModelTC/LightX2V.git
cd LightX2V
git reset --hard 2f952497d4c1fe94dd4555fc7abdce9ef0a473d1

cp ../init_vid_gen_venv.sh ../vid_gen.py ./

conda create -n vid_gen python=3.10 -y
conda activate vid_gen
./init_vid_gen_venv.sh
```

Для генерации видео была задействована модель `HunyanVideo`.
Сначал - скачайте модель с HuggingFace.
Затем - запустите генерацию видео.

```shell
hf download tencent/HunyuanVideo-1.5

python vid_gen.py --model_path "/path/to/HunyuanVideo-1.5" --json_path "MVBench_emb.json" --save_folder "./generated_videos/" --img_folder "./generated_images/" --dataset_type "mv"
```

# Дообучение

## Подготовка набора данных
Создайте набор данных для SFT.

```shell
python generate_ft_json.py --input_json_path "MVBench_emb.json" --video_path "./generated_videos/"
```

Установите LlamaFactory.
Запустите обучение:
```shell
llamafactory-cli train minicpmv4_5_lora.yaml
```

Если вылезет ошибка с `model.tp_plan is None` внутри выражения `re.compile`, измените исходный код, добавив строчку с `model.tp_plan = {}` перед строчкой с ошибкой.

Смёрджите адаптеры
```shell
llamafactory-cli export configs/minicpmv4_5_lora_export.yaml
```

Убедитесь, что путь до модели в конфиге указан правильно.

Если возникает ошибка с инициализацией процессора, связанная с `chat_template`, добавьте `**kwargs` к __init__ методу для файла /path/to/model/modeling_processor.py класса MiniCPMVProcessor