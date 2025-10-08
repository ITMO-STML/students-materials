FROM python:3.9-slim

WORKDIR /app

COPY classifier.py .
COPY requirements.txt .

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118

RUN pip install --no-cache-dir pandas tqdm opencv-python pybind11>=2.12


CMD ["python", "classifier.py", "--path", "/data", "--batch_size", "32", "--epochs", "50", "--lr", "0.001"]
