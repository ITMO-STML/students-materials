import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
import os

print("="*60)
print("📊 СОЗДАНИЕ ВИЗУАЛИЗАЦИЙ ДЛЯ ТЕСТА")
print("="*60)

# Загружаем данные
train_df = pd.read_csv('data/raw/wine_variations.csv')
test_df = pd.read_csv('data/raw/winequality_old.csv')

# Обучаем модель
X_train = train_df.drop('quality', axis=1)
if 'Id' in X_train.columns:
    X_train = X_train.drop('Id', axis=1)
y_train = train_df['quality']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
model = GaussianNB()
model.fit(X_train_scaled, y_train)

# Тестируем
X_test = test_df.drop('quality', axis=1)
if 'Id' in X_test.columns:
    X_test = X_test.drop('Id', axis=1)
y_test = test_df['quality']
X_test_scaled = scaler.transform(X_test)
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)

# Создаем папку для визуализаций
os.makedirs('reports/test_visualizations', exist_ok=True)

print("\n[1/6] Создание визуализаций...")

# 1. Сравнение истинных и предсказанных значений
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# График 1: Точность по классам
ax1 = axes[0, 0]
classes = sorted(set(y_test))
accuracy_by_class = []
for cls in classes:
    mask = y_test == cls
    if mask.sum() > 0:
        acc = (y_pred[mask] == y_test[mask]).mean()
        accuracy_by_class.append(acc)
    else:
        accuracy_by_class.append(0)

colors = ['green' if acc > 0.5 else 'orange' if acc > 0 else 'red' for acc in accuracy_by_class]
bars = ax1.bar(classes, accuracy_by_class, color=colors, edgecolor='black')
ax1.set_xlabel('Качество вина', fontsize=12)
ax1.set_ylabel('Точность', fontsize=12)
ax1.set_title('Точность предсказания по каждому качеству', fontsize=14)
ax1.set_ylim(0, 1)
ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
for bar, acc in zip(bars, accuracy_by_class):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{acc:.0%}', ha='center', fontsize=10)

# График 2: Сравнение распределений
ax2 = axes[0, 1]
true_dist = pd.Series(y_test).value_counts().sort_index()
pred_dist = pd.Series(y_pred).value_counts().sort_index()
x = np.arange(len(classes))
width = 0.35
ax2.bar(x - width/2, [true_dist.get(c, 0) for c in classes], width, 
        label='Истинные', color='blue', alpha=0.7)
ax2.bar(x + width/2, [pred_dist.get(c, 0) for c in classes], width, 
        label='Предсказанные', color='orange', alpha=0.7)
ax2.set_xlabel('Качество вина', fontsize=12)
ax2.set_ylabel('Количество', fontsize=12)
ax2.set_title('Распределение: Истинные vs Предсказанные', fontsize=14)
ax2.set_xticks(x)
ax2.set_xticklabels(classes)
ax2.legend()

# График 3: Линия сравнения
ax3 = axes[1, 0]
ax3.plot(range(len(test_df)), y_test, 'o-', label='Истинное', color='blue', 
         markersize=8, linewidth=2)
ax3.plot(range(len(test_df)), y_pred, 's-', label='Предсказанное', color='red', 
         markersize=8, linewidth=2)
ax3.set_xlabel('Образец вина', fontsize=12)
ax3.set_ylabel('Качество', fontsize=12)
ax3.set_title('Сравнение истинных и предсказанных значений', fontsize=14)
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_xticks(range(len(test_df)))
ax3.set_xticklabels([f'#{i+1}' for i in range(len(test_df))], rotation=45)

# График 4: Уверенность предсказаний
ax4 = axes[1, 1]
confidence = np.max(y_proba, axis=1)
colors = ['green' if p == t else 'red' for p, t in zip(y_pred, y_test)]
bars = ax4.bar(range(len(test_df)), confidence, color=colors, edgecolor='black', alpha=0.7)
ax4.set_xlabel('Образец вина', fontsize=12)
ax4.set_ylabel('Уверенность', fontsize=12)
ax4.set_title('Уверенность модели (зеленый = правильно, красный = ошибка)', fontsize=12)
ax4.set_ylim(0, 1)
ax4.set_xticks(range(len(test_df)))
ax4.set_xticklabels([f'#{i+1}' for i in range(len(test_df))], rotation=45)
ax4.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('reports/test_visualizations/comparison_plots.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ comparison_plots.png")

# 2. Матрица ошибок
plt.figure(figsize=(10, 8))
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_test, y_pred, labels=sorted(set(y_test) | set(y_pred)))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='YlOrRd', 
            xticklabels=sorted(set(y_test) | set(y_pred)),
            yticklabels=sorted(set(y_test) | set(y_pred)))
plt.title('Матрица ошибок', fontsize=16)
plt.ylabel('Истинное качество', fontsize=12)
plt.xlabel('Предсказанное качество', fontsize=12)
plt.tight_layout()
plt.savefig('reports/test_visualizations/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ confusion_matrix.png")

# 3. Зависимость от ключевых признаков
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Алкоголь vs Качество
ax1 = axes[0]
for i, (true, pred) in enumerate(zip(y_test, y_pred)):
    color = 'green' if true == pred else 'red'
    ax1.scatter(test_df.iloc[i]['alcohol'], true, c=color, s=100, alpha=0.7, edgecolors='black')
ax1.set_xlabel('Алкоголь (%)', fontsize=12)
ax1.set_ylabel('Истинное качество', fontsize=12)
ax1.set_title('Алкоголь vs Качество\n(зеленый=правильно, красный=ошибка)', fontsize=12)
ax1.grid(True, alpha=0.3)

# Летучая кислотность vs Качество
ax2 = axes[1]
for i, (true, pred) in enumerate(zip(y_test, y_pred)):
    color = 'green' if true == pred else 'red'
    ax2.scatter(test_df.iloc[i]['volatile acidity'], true, c=color, s=100, alpha=0.7, edgecolors='black')
ax2.set_xlabel('Летучая кислотность', fontsize=12)
ax2.set_ylabel('Истинное качество', fontsize=12)
ax2.set_title('Кислотность vs Качество\n(зеленый=правильно, красный=ошибка)', fontsize=12)
ax2.grid(True, alpha=0.3)

# Сульфаты vs Качество
ax3 = axes[2]
for i, (true, pred) in enumerate(zip(y_test, y_pred)):
    color = 'green' if true == pred else 'red'
    ax3.scatter(test_df.iloc[i]['sulphates'], true, c=color, s=100, alpha=0.7, edgecolors='black')
ax3.set_xlabel('Сульфаты', fontsize=12)
ax3.set_ylabel('Истинное качество', fontsize=12)
ax3.set_title('Сульфаты vs Качество\n(зеленый=правильно, красный=ошибка)', fontsize=12)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('reports/test_visualizations/features_vs_quality.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ features_vs_quality.png")

# 4. Тепловая карта корреляций
plt.figure(figsize=(12, 10))
correlation_matrix = test_df.corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=0.5)
plt.title('Корреляция признаков (тестовые данные)', fontsize=16)
plt.tight_layout()
plt.savefig('reports/test_visualizations/correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ correlation_heatmap.png")

# 5. Детальная таблица результатов
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('tight')
ax.axis('off')

# Создаем таблицу
table_data = []
headers = ['№', 'Алкоголь', 'Кислотность', 'Сульфаты', 'pH', 'Истинное', 'Предсказанное', 'Уверенность', 'Результат']

for i in range(len(test_df)):
    confidence = np.max(y_proba[i]) * 100
    verdict = '✓' if y_test.iloc[i] == y_pred[i] else '✗'
    table_data.append([
        i+1,
        f"{test_df.iloc[i]['alcohol']:.1f}",
        f"{test_df.iloc[i]['volatile acidity']:.2f}",
        f"{test_df.iloc[i]['sulphates']:.2f}",
        f"{test_df.iloc[i]['pH']:.2f}",
        y_test.iloc[i],
        y_pred[i],
        f"{confidence:.0f}%",
        verdict
    ])

# Цветовая кодировка для результатов
colors = [['#d4edda' if row[-1] == '✓' else '#f8d7da' for _ in range(len(headers))] for row in table_data]

table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center',
                 cellColours=colors)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.5)

plt.title('Детальные результаты тестирования', fontsize=14, pad=20)
plt.tight_layout()
plt.savefig('reports/test_visualizations/detailed_results_table.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ detailed_results_table.png")

# 6. Box plot сравнения
fig, ax = plt.subplots(figsize=(10, 6))
data_to_plot = []
labels = []
for i in range(len(test_df)):
    data_to_plot.append([y_test.iloc[i], y_pred[i]])
    labels.append(f'#{i+1}')

positions = np.arange(len(test_df))
width = 0.35

ax.bar(positions - width/2, y_test, width, label='Истинное', color='blue', alpha=0.7)
ax.bar(positions + width/2, y_pred, width, label='Предсказанное', color='orange', alpha=0.7)
ax.set_xlabel('Образец вина', fontsize=12)
ax.set_ylabel('Качество', fontsize=12)
ax.set_title('Сравнение истинных и предсказанных значений', fontsize=14)
ax.set_xticks(positions)
ax.set_xticklabels(labels, rotation=45)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('reports/test_visualizations/bar_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ bar_comparison.png")

print("\n" + "="*60)
print("✅ ВСЕ ВИЗУАЛИЗАЦИИ СОЗДАНЫ!")
print("="*60)
print("\n📁 Папка с визуализациями: reports/test_visualizations/")
print("Файлы:")
print("  📊 comparison_plots.png      - 4 графика сравнения")
print("  📊 confusion_matrix.png      - матрица ошибок")
print("  📊 features_vs_quality.png   - зависимость от признаков")
print("  📊 correlation_heatmap.png   - корреляция признаков")
print("  📊 detailed_results_table.png - таблица результатов")
print("  📊 bar_comparison.png        - столбчатая диаграмма")

# Открываем папку
print("\n📂 Открыть папку с визуализациями:")
print("   explorer.exe reports/test_visualizations/")
