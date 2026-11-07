FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV TORCH_HOME=/app/.cache/torch

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model.py data.py train.py inference.py eda.py report.py ./ 

CMD ["python", "train.py"]
