import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("="*60)
print("🍷 ТЕСТИРОВАНИЕ: Обучено на 40, тестируем на 11")
print("="*60)

# 1. Загружаем 40 образцов для обучения
print("\n[1/4] Загрузка 40 образцов для обучения...")
train_df = pd.read_csv('data/raw/wine_variations.csv')
if 'Id' in train_df.columns:
    train_df = train_df.drop('Id', axis=1)
print(f"✅ Обучающая выборка: {len(train_df)} образцов")

# 2. Загружаем 11 старых образцов для тестирования
print("\n[2/4] Загрузка 11 образцов для тестирования...")
test_df = pd.read_csv('data/raw/winequality_old.csv')
if 'Id' in test_df.columns:
    test_df = test_df.drop('Id', axis=1)
print(f"✅ Тестовая выборка: {len(test_df)} образцов")

# 3. Обучаем модель на 40 образцах
print("\n[3/4] Обучение модели...")
X_train = train_df.drop('quality', axis=1).values
y_train = train_df['quality'].values
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
model = GaussianNB()
model.fit(X_train_scaled, y_train)
print("✅ Модель обучена")

# 4. Тестируем
print("\n[4/4] Тестирование...")
X_test = test_df.drop('quality', axis=1).values
y_test = test_df['quality'].values
X_test_scaled = scaler.transform(X_test)
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n{'='*50}")
print(f"🎯 ТОЧНОСТЬ: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"{'='*50}")

# Подробные результаты
print("\n📋 РЕЗУЛЬТАТЫ ПО КАЖДОМУ ВИНУ:")
print("-"*60)
print(f"{'Алкоголь':<10} {'Кислотность':<12} {'Истинное':<10} {'Предсказанное':<13} {'Результат'}")
print("-"*60)

for i in range(len(test_df)):
    alcohol = test_df.iloc[i]['alcohol']
    volatility = test_df.iloc[i]['volatile acidity']
    true_q = y_test[i]
    pred_q = y_pred[i]
    verdict = "✅" if true_q == pred_q else "❌"
    print(f"{alcohol:<10.1f} {volatility:<12.2f} {true_q:<10} {pred_q:<13} {verdict}")

print(f"\n✅ Правильных: {(y_pred == y_test).sum()} из {len(y_test)}")
print(f"❌ Ошибок: {(y_pred != y_test).sum()} из {len(y_test)}")
print("="*60)
