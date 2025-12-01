FROM python:3.9-slim

WORKDIR /app

COPY rl_policy_gradient.py .
COPY requirements.txt .

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA 11.8 support for GTX 1050 Ti
RUN pip install --no-cache-dir torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies (excluding torch from requirements.txt)
RUN pip install --no-cache-dir numpy>=1.24.0 gymnasium>=0.29.0 matplotlib>=3.7.0 tqdm>=4.65.0

CMD ["python", "rl_policy_gradient.py"]

