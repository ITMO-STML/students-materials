FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Кэш моделей до скачивания
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV TORCH_HOME=/app/.cache/torch

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Скачиваем модели в кэш
RUN python -c "from transformers import AutoTokenizer, AutoModel; \
    AutoTokenizer.from_pretrained('distilbert-base-uncased'); \
    AutoModel.from_pretrained('distilbert-base-uncased')"

RUN python -c "from torchvision.models import resnet18, ResNet18_Weights; \
    resnet18(weights=ResNet18_Weights.DEFAULT)"

COPY model.py data.py train.py inference.py ./

CMD ["python", "train.py"]
