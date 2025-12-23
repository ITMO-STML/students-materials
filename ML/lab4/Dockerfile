FROM python:3.9-slim

WORKDIR /app

COPY controlnet_diffusion.py .
COPY requirements.txt .

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем PyTorch с CUDA поддержкой
# Используем последние версии для поддержки новых видеокарт
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 || \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 || \
    pip install --no-cache-dir torch torchvision torchaudio

# Устанавливаем остальные зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Пытаемся установить xformers (опционально, может не работать без CUDA)
RUN pip install --no-cache-dir xformers || echo "xformers не установлен, будет использовано стандартное внимание"

CMD ["python", "controlnet_diffusion.py", "--help"]
