#!/bin/bash

# Скрипт для запуска демонстрационной программы

# Проверяем наличие виртуального окружения
if [ -d "../.venv" ]; then
    echo "Активация виртуального окружения..."
    source ../.venv/bin/activate
fi

# Устанавливаем путь к модели (можно изменить)
export MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-../checkpoints_cnn_final/best_cnn_final.pt}"

# Проверяем наличие модели
if [ ! -f "$MODEL_CHECKPOINT" ]; then
    echo "Внимание: Модель не найдена по пути: $MODEL_CHECKPOINT"
    echo "Установите переменную окружения MODEL_CHECKPOINT или поместите модель в указанный путь."
    echo ""
    echo "Пример:"
    echo "  export MODEL_CHECKPOINT=../checkpoints_cnn_hyperopt_v2/best_cnn_hyperopt.pt"
    echo "  python app.py"
    echo ""
    read -p "Продолжить все равно? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Запускаем приложение
echo "Запуск демонстрационной программы..."
echo "Модель: $MODEL_CHECKPOINT"
echo ""
python app.py
