# Лаба 1: Промежуточное фузирование модальностей

## Задача
Классификация товаров по изображению + текстовому описанию.
10 категорий, промежуточное фузирование через cross-attention.

## Данные
- `data/data.csv` — 10k товаров с полями image, description, category
- `data/*.jpg` — изображения товаров
- В обучении: топ-10 категорий, до 500 примеров на класс, разбиение 70/15/15

## Архитектура
CLSCrossAttentionFusion:
- ResNet18 (заморожен) → image embedding 512
- DistilBERT (заморожен) → text tokens 768
- Linear projection 512→768 → Query
- Cross-attention: Q=image, K=V=text tokens
- LayerNorm → MLP classifier → 10 классов

## Результаты
- Best smoothed val: 0.9587 @ epoch 12
- **Test accuracy: 0.9440**
- Обучение: NVIDIA GTX 1080 Ti, ~6 минут

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