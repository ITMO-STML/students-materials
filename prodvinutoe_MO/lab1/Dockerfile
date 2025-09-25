FROM python:3.13-slim

# Обновляем pip и инструменты сборки
RUN pip install --upgrade pip setuptools wheel

# Создаём рабочую директорию
WORKDIR /app

# Копируем только requirements.txt (для кеша)
COPY requirements.txt /app/

RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем проект (с учётом .dockerignore)
COPY . /app

# Открываем порт для Jupyter
EXPOSE 8888

# Запускаем Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=", "--NotebookApp.password="]
