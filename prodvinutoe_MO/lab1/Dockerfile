FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Устанавливаем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем проект (data/models/kaggle.json исключены .dockerignore)
COPY . .

# Создаём директории и kaggle config dir (файл туда будем монтировать)
RUN mkdir -p /app/data /app/models /root/.kaggle && chmod 700 /root/.kaggle

ENV KAGGLE_CONFIG_DIR=/root/.kaggle

RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["bash", "/app/entrypoint.sh"]
