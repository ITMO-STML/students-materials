# Лаба 1: Промежуточное фузирование модальностей

## Задача
Классификация товаров по категориям и полу по изображению + текстовому описанию.
10 категорий, промежуточное фузирование через cross-attention.

## Данные
- `data/data.csv` — 10k товаров с полями image, description, category
- `data/*.jpg` — изображения товаров
- В обучении: топ-10 категорий, до 500 примеров на класс, разбиение 70/15/15

## Запуск через Docker
\```bash
docker build -t lab-fusion .
docker run --rm --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  lab-fusion
\```

## Инференс
\```bash
docker run --rm --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  lab-fusion python inference.py
\```

## Файлы
- `model.py` — архитектура CLS Cross-Attention
- `data.py` — Dataset, transforms
- `train.py` — обучение с early stopping, cosine annealing
- `inference.py` — предсказание
- `Dockerfile` — воспроизводимая сборка