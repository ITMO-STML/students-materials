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

EMB_DIR = "/home/ext-ivanova-mk@ad.speechpro.com/test_dir/tlm/feature_extraction/nnces_titanet_embeddings"
MODEL_PATH = "/home/ext-ivanova-mk@ad.speechpro.com/test_dir/tlm/models/tlm_age_mixup_3.pkl"
OUTPUT_DIR = EMB_DIR

print("\n1. Загрузка эмбеддингов...")
all_emb = np.load(os.path.join(EMB_DIR, "all_embeddings_nnces.npy"))
print(f"   Эмбеддинги загружены, форма: {all_emb.shape}")

if len(all_emb.shape) == 3 and all_emb.shape[1] == 1:
    all_emb = all_emb.squeeze(1)

print("\n2. Загрузка метаданных...")
df_meta = pd.read_csv(os.path.join(EMB_DIR, "nnces_titanet_metadata_valid.csv"))
print(f"   Метаданные загружены, {len(df_meta)} записей")

if len(df_meta) != len(all_emb):
    print(f"   Метаданных: {len(df_meta)}, Эмбеддингов: {len(all_emb)}")
    min_len = min(len(df_meta), len(all_emb))
    all_emb = all_emb[:min_len]
    df_meta = df_meta.iloc[:min_len]
    print(f"   Обрезано до {min_len} записей")

print("\n3. Загрузка модели TLM...")
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)
print(f"   Модель загружена, тип: {type(model)}")

print("\n4. Выполнение предсказаний...")
predictions = model.predict(all_emb)

print("\n5. Сохранение результатов...")
df_meta['age_pred'] = predictions

output_csv = os.path.join(OUTPUT_DIR, "nnces_titanet_metadata_with_predictions.csv")
df_meta.to_csv(output_csv, index=False)
print(f"   Результат сохранён в {output_csv}")

np.save(os.path.join(OUTPUT_DIR, "age_predictions_nnces.npy"), predictions)
print(f"   Предсказания сохранены в age_predictions_nnces.npy")

output_txt = os.path.join(OUTPUT_DIR, "nnces_age_predictions.txt")
with open(output_txt, 'w', encoding='utf-8') as f:
    for idx, row in df_meta.iterrows():
        rel_path = row.get('rel_path', row.get('file_path', f"file_{idx}"))
        f.write(f"{rel_path} {row['age_pred']:.2f}\n")
print(f"   Текстовый файл с предсказаниями сохранён в {output_txt}")

if 'age' in df_meta.columns:
    print("\n7. Сравнение с истинными возрастами:")
    true_ages = df_meta['age'].values
    print(f"   Истинные - Min: {true_ages.min():.1f}, Max: {true_ages.max():.1f}, Mean: {true_ages.mean():.1f}")
    print(f"   Предсказанные - Min: {predictions.min():.1f}, Max: {predictions.max():.1f}, Mean: {predictions.mean():.1f}")

print("\n8. Первые 10 предсказаний:")
print(predictions[:10])