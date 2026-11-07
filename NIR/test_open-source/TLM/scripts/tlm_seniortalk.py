import pickle
import sys
import os
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tlm import TLM, Node
from baselines import BaselineModels, MLPRegressor

class NRT:
    def __init__(self):
        pass

# Пути
EMB_DIR = "/home/ext-ivanova-mk@ad.speechpro.com/test_dir/tlm/feature_extraction/seniortalk_titanet_embeddings"
MODEL_PATH = "/home/ext-ivanova-mk@ad.speechpro.com/test_dir/tlm/models/tlm_age_mixup_3.pkl"
OUTPUT_DIR = EMB_DIR

print("\n1. Загрузка эмбеддингов...")
all_emb = np.load(os.path.join(EMB_DIR, "all_embeddings_seniortalk.npy"))
print(f"   Эмбеддинги загружены, форма: {all_emb.shape}")

if len(all_emb.shape) == 3 and all_emb.shape[1] == 1:
    all_emb = all_emb.squeeze(1)

print("\n2. Загрузка метаданных...")
df_meta = pd.read_csv(os.path.join(EMB_DIR, "seniortalk_titanet_metadata.csv"))
print(f"   Метаданные загружены, {len(df_meta)} записей")

# Проверяем соответствие размеров
if len(df_meta) != len(all_emb):
    print(f"   ВНИМАНИЕ: Несоответствие размеров!")
    print(f"   Метаданных: {len(df_meta)}, Эмбеддингов: {len(all_emb)}")
    min_len = min(len(df_meta), len(all_emb))
    all_emb = all_emb[:min_len]
    df_meta = df_meta.iloc[:min_len]
    print(f"   Обрезано до {min_len} записей")

print("\n3. Загрузка модели TLM...")
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

print("\n4. Выполнение предсказаний...")
predictions = model.predict(all_emb)
print(f"   Предсказания получены, форма: {predictions.shape}")

print("\n5. Сохранение результатов...")
df_meta['age_pred'] = predictions

# Сохраняем результат
output_csv = os.path.join(OUTPUT_DIR, "seniortalk_titanet_metadata_with_predictions.csv")
df_meta.to_csv(output_csv, index=False)
print(f"   Результат сохранён в {output_csv}")

# Сохраняем предсказания в отдельный файл
np.save(os.path.join(OUTPUT_DIR, "age_predictions_seniortalk.npy"), predictions)
print(f"   Предсказания сохранены в age_predictions_seniortalk.npy")

# Сохраняем файл с предсказаниями в формате для оценки
output_txt = os.path.join(OUTPUT_DIR, "seniortalk_age_predictions.txt")
with open(output_txt, 'w', encoding='utf-8') as f:
    for idx, row in df_meta.iterrows():
        f.write(f"{row['file_name']} {row['age_pred']:.2f}\n")
print(f"   Текстовый файл с предсказаниями сохранён в {output_txt}")

print("\n7. Первые 10 предсказаний:")
print(predictions[:10])