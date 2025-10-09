#!/bin/bash
set -e  # если какая-то команда завершится с ошибкой — скрипт сразу остановится

DATA_FILE="data/pneumonia.csv"
MODEL_FILE="models/model_weights.pth"

if [ -f "$DATA_FILE" ] && [ -f "$MODEL_FILE" ]; then
    echo "✅ Найдены данные и модель. Запускаем инференс..."

    # Выполняем ноутбук
    jupyter nbconvert --to notebook --execute /app/script/inference.ipynb \
        --output /app/script/inference_output.ipynb --allow-errors

    # Выводим результат последней ячейки в консоль
    if [ -f /app/script/last_output.txt ]; then
        cat /app/script/last_output.txt
    else
        echo "⚠️ last_output.txt не найден. Проверь последнюю ячейку ноутбука."
    fi

else
    echo "⚙️ Отсутствуют данные или модель. Сначала скачаем и обучим."

    mkdir -p data models

    echo "⬇️ Скачиваем данные с Kaggle..."
    python script/download_data.py

    echo "🎓 Обучаем модель..."
    jupyter nbconvert --to notebook --execute /app/script/train.ipynb \
        --output /app/script/train_output.ipynb --allow-errors

    echo "🚀 Запускаем инференс..."
    jupyter nbconvert --to notebook --execute /app/script/inference.ipynb \
        --output /app/script/inference_output.ipynb --allow-errors

    # Вывод результата последней ячейки
    if [ -f /app/script/last_output.txt ]; then
        cat /app/script/last_output.txt
    else
        echo "⚠️ last_output.txt не найден. Проверь последнюю ячейку ноутбука."
    fi
fi

echo "🏁 Всё готово!"
