import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("="*60)
print("🍷 ТЕСТИРОВАНИЕ МОДЕЛИ НА НОВЫХ ДАННЫХ")
print("="*60)

# 1. Загружаем старые данные (обучающая выборка)
print("\n[1/4] Загрузка старых данных (обучение)...")
try:
    old_df = pd.read_csv('data/raw/winequality.csv')
    print(f"✅ Загружены старые данные: {len(old_df)} образцов")
except:
    # Если старый файл перезаписан, используем wine_variations как старый
    old_df = pd.read_csv('data/raw/wine_variations.csv')
    print(f"⚠️ Используем wine_variations как старые данные: {len(old_df)} образцов")

# Удаляем Id если есть
if 'Id' in old_df.columns:
    old_df = old_df.drop('Id', axis=1)

print(f"📊 Старые данные - качество: {old_df['quality'].min()}-{old_df['quality'].max()}")

# 2. Загружаем новые данные (тестовая выборка)
print("\n[2/4] Загрузка новых данных (тестирование)...")
new_df = pd.read_csv('data/raw/wine_variations.csv')
if 'Id' in new_df.columns:
    new_df = new_df.drop('Id', axis=1)
print(f"✅ Загружены новые данные: {len(new_df)} образцов")
print(f"📊 Новые данные - качество: {new_df['quality'].min()}-{new_df['quality'].max()}")

# 3. Обучаем модель на старых данных
print("\n[3/4] Обучение модели на старых данных...")

# Подготовка данных для обучения
X_old = old_df.drop('quality', axis=1).values
y_old = old_df['quality'].values

# Масштабируем
scaler = StandardScaler()
X_old_scaled = scaler.fit_transform(X_old)

# Обучаем модель
model = GaussianNB()
model.fit(X_old_scaled, y_old)
print("✅ Модель обучена")

# 4. Тестируем на новых данных
print("\n[4/4] Тестирование на новых данных...")

# Подготавливаем новые данные (используем тот же scaler!)
X_new = new_df.drop('quality', axis=1).values
y_new = new_df['quality'].values
X_new_scaled = scaler.transform(X_new)  # Важно: transform, а не fit_transform!

# Предсказываем
y_pred = model.predict(X_new_scaled)

# Оцениваем
accuracy = accuracy_score(y_new, y_pred)

print(f"\n{'='*50}")
print(f"🎯 ТОЧНОСТЬ НА НОВЫХ ДАННЫХ: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"{'='*50}")

# Сравнение распределения
print("\n📊 СРАВНЕНИЕ РАСПРЕДЕЛЕНИЯ КАЧЕСТВА:")
print("\nСтарые данные (обучение):")
old_dist = old_df['quality'].value_counts().sort_index()
for q, count in old_dist.items():
    print(f"  Качество {q}: {count:2d} образцов")

print("\nНовые данные (тестирование):")
new_dist = new_df['quality'].value_counts().sort_index()
for q, count in new_dist.items():
    print(f"  Качество {q}: {count:2d} образцов")

print("\n📈 ОТЧЕТ ПО КЛАССИФИКАЦИИ (на новых данных):")
print(classification_report(y_new, y_pred, zero_division=0))

# Матрица ошибок
conf_matrix = confusion_matrix(y_new, y_pred)

# Создаем визуализации
os.makedirs('reports', exist_ok=True)

# 1. Матрица ошибок
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Матрица ошибок - Тестирование на новых данных')
plt.ylabel('Истинное качество')
plt.xlabel('Предсказанное качество')
plt.savefig('reports/test_on_new_data_confusion.png', dpi=100, bbox_inches='tight')
plt.close()
print("\n✅ Сохранено: reports/test_on_new_data_confusion.png")

# 2. Сравнение предсказаний
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.bar(new_dist.index, new_dist.values, alpha=0.7, label='Реальные', color='blue')
pred_dist = pd.Series(y_pred).value_counts().sort_index()
plt.bar(pred_dist.index, pred_dist.values, alpha=0.7, label='Предсказанные', color='orange')
plt.xlabel('Качество')
plt.ylabel('Количество')
plt.title('Реальные vs Предсказанные')
plt.legend()
plt.xticks(range(4, 9))

plt.subplot(1, 2, 2)
# Точность по каждому качеству
accuracy_by_quality = []
for q in sorted(set(y_new)):
    mask = y_new == q
    if mask.sum() > 0:
        acc = (y_pred[mask] == y_new[mask]).mean()
        accuracy_by_quality.append(acc)
    else:
        accuracy_by_quality.append(0)

plt.bar(sorted(set(y_new)), accuracy_by_quality, color='green')
plt.xlabel('Качество')
plt.ylabel('Точность')
plt.title('Точность предсказания по каждому качеству')
plt.ylim(0, 1)
plt.xticks(range(4, 9))

plt.tight_layout()
plt.savefig('reports/test_on_new_data_comparison.png', dpi=100, bbox_inches='tight')
plt.close()
print("✅ Сохранено: reports/test_on_new_data_comparison.png")

# Сохраняем отчет
with open('reports/test_on_new_data_report.txt', 'w', encoding='utf-8') as f:
    f.write("="*60 + "\n")
    f.write("ОТЧЕТ: ТЕСТИРОВАНИЕ МОДЕЛИ НА НОВЫХ ДАННЫХ\n")
    f.write("="*60 + "\n\n")
    f.write(f"Обучено на: {len(old_df)} образцов\n")
    f.write(f"Протестировано на: {len(new_df)} образцов\n\n")
    f.write(f"Точность: {accuracy:.4f} ({accuracy*100:.2f}%)\n\n")
    f.write("Распределение качества в обучающих данных:\n")
    for q, count in old_dist.items():
        f.write(f"  Качество {q}: {count} образцов\n")
    f.write("\nРаспределение качества в тестовых данных:\n")
    for q, count in new_dist.items():
        f.write(f"  Качество {q}: {count} образцов\n")
    f.write("\n" + "-"*40 + "\n")
    f.write("ОТЧЕТ ПО КЛАССИФИКАЦИИ:\n")
    f.write("-"*40 + "\n")
    f.write(classification_report(y_new, y_pred, zero_division=0))
    f.write("\n\nМАТРИЦА ОШИБОК:\n")
    f.write("-"*40 + "\n")
    f.write(str(conf_matrix))

print("\n✅ Сохранено: reports/test_on_new_data_report.txt")

# Выводим примеры ошибок
print("\n🔍 ПРИМЕРЫ ОШИБОК МОДЕЛИ:")
errors = y_new != y_pred
error_indices = np.where(errors)[0]

if len(error_indices) > 0:
    print(f"\nНайдено {len(error_indices)} ошибок. Первые 5:")
    for i, idx in enumerate(error_indices[:5]):
        print(f"\n  Образец {idx+1}:")
        print(f"    Истинное качество: {y_new[idx]}")
        print(f"    Предсказанное: {y_pred[idx]}")
        print(f"    Алкоголь: {new_df.iloc[idx]['alcohol']:.1f}%")
        print(f"    Кислотность: {new_df.iloc[idx]['volatile acidity']:.2f}")
else:
    print("\n🎉 Ошибок нет! Модель идеально предсказала все образцы!")

print("\n" + "="*60)
print("📁 Результаты в папке reports/")
print("  - test_on_new_data_confusion.png")
print("  - test_on_new_data_comparison.png")
print("  - test_on_new_data_report.txt")
print("="*60)
