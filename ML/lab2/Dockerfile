FROM python:3.9-slim

WORKDIR /app
ENV VIDEO_PATH=/videos


COPY video_classifier.py .

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118

RUN pip install --no-cache-dir tqdm datasets opencv-python pybind11>=2.12 scikit-learn numpy

CMD ["python", "video_classifier.py"]