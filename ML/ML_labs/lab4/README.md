# Lab 4 - DDPM на MNIST (class-conditional)

Реализация генеративной модели диффузии (Denoising Diffusion Probabilistic
Model, DDPM) на датасете MNIST. Модель **class-conditional**: на вход
подаётся целевая метка класса, что позволяет генерировать цифры заданного
класса и осмысленно оценивать качество.

## Структура

```
lab4/
├── data/                    # MNIST (скачивается автоматически, монтируется)
├── outputs/                 # артефакты (монтируется)
│   ├── checkpoint.pt        # веса DDPM
│   ├── classifier.pt        # веса оценщика
│   ├── generated.pt         # сгенерированные изображения
│   ├── loss.csv             # кривая обучения
│   ├── classification_report.txt
│   ├── confusion_matrix.csv
│   └── report.html          # итоговый HTML-отчёт
├── src/
│   ├── data.py              # загрузка MNIST, DataLoader'ы
│   ├── model.py             # U-Net (class-conditional)
│   ├── diffusion.py         # forward/reverse процесс DDPM
│   ├── train.py             # оркестратор: DDPM - evaluate - report
│   ├── evaluate.py          # классификатор + генерация + метрики
│   └── report.py            # сборка report.html
├── Dockerfile
├── requirements.txt
├── .dockerignore
├── .gitignore
└── README.md
```

## Запуск

Перед первым запуском создайте папки на хосте:

```bash
mkdir -p data outputs
```

Сборка образа:

```bash
sudo docker build -t lab4-diffusion .
```

Запуск:

```bash
sudo docker run --rm --gpus all \
  -v "$(pwd)/outputs:/app/outputs" \
  -v "$(pwd)/data:/app/data" \
  --user "$(id -u):$(id -g)" \
  lab4-diffusion
```

Результаты:
- `outputs/report.html` - итоговый отчёт;
- `outputs/checkpoint.pt` - веса обученной DDPM;
- `outputs/classification_report.txt`, `outputs/confusion_matrix.csv` - метрики.

## Результаты

- **DDPM:** MSE loss снизился с 0.107 до 0.026 за 12 эпох.
- **Классификатор-оценщик:** accuracy 98.8% на тесте MNIST.
- **Качество генерации:** accuracy 78.1% (сгенерированные изображения правильно классифицируются независимой CNN). Наиболее трудные классы - 3, 7, 9 (в MNIST они визуально близки).