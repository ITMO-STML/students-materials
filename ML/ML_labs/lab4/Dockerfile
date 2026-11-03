FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV MPLCONFIGDIR=/tmp/matplotlib

COPY src/ ./src/

RUN mkdir -p /app/outputs /app/data

CMD ["python", "src/train.py"]