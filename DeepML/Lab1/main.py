import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import os


# 1. Загрузка данных
print("\n[1/5] Загрузка данных...")

# Проверяем, существует ли файл с данными
if os.path.exists('data/raw/winequality.csv'):
    df = pd.read_csv('data/raw/winequality.csv')
    print(f"Данные загружены из data/raw/winequality.csv")
else:
    print("❌ Файл data/raw/winequality.csv не найден!")
    data = {
        'fixed acidity': [7.4, 7.8, 7.8, 11.2, 7.4],
        'volatile acidity': [0.7, 0.88, 0.76, 0.28, 0.7],
        'citric acid': [0.0, 0.0, 0.04, 0.56, 0.0],
        'residual sugar': [1.9, 2.6, 2.3, 1.9, 1.9],
        'chlorides': [0.076, 0.098, 0.092, 0.075, 0.076],
        'free sulfur dioxide': [11.0, 25.0, 15.0, 17.0, 11.0],
        'total sulfur dioxide': [34.0, 67.0, 54.0, 60.0, 34.0],
        'density': [0.9978, 0.9968, 0.997, 0.998, 0.9978],
        'pH': [3.51, 3.2, 3.26, 3.16, 3.51],
        'sulphates': [0.56, 0.68, 0.65, 0.58, 0.56],
        'alcohol': [9.4, 9.8, 9.8, 9.8, 9.4],
        'quality': [5, 5, 5, 6, 5]
    }
    df = pd.DataFrame(data)

# Удаляем Id если есть
if 'Id' in df.columns:
    df = df.drop('Id', axis=1)

# 2. Подготовка данных
print("\n[2/5] Подготовка данных...")
X = df.drop('quality', axis=1).values
y = df['quality'].values

# Масштабируем
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Разделяем на train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"Train: {len(X_train)} образцов")
print(f"Test: {len(X_test)} образцов")

# 3. Обучение
print("\n[3/5] Обучение модели...")
model = GaussianNB()
model.fit(X_train, y_train)

print("Обучение завершено")
