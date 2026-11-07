import pickle
import sys
import os
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class NRT:
    def __init__(self):
        pass

from tlm import TLM, Node
from baselines import BaselineModels, MLPRegressor

EMB_DIR = "/home/ext-ivanova-mk@ad.speechpro.com/test_dir/tlm/feature_extraction/agevox_titanet_embeddings"
MODEL_PATH = "/home/ext-ivanova-mk@ad.speechpro.com/test_dir/tlm/models/tlm_age_mixup_3.pkl"
OUTPUT_DIR = EMB_DIR

print("\n1. Загрузка эмбеддингов...")
all_emb = np.load(os.path.join(EMB_DIR, "all_embeddings_agevoxceleb.npy"))
print(f"   Эмбеддинги загружены, форма: {all_emb.shape}")

if len(all_emb.shape) == 3 and all_emb.shape[1] == 1:
    all_emb = all_emb.squeeze(1)

print("\n2. Загрузка метаданных...")
df_meta = pd.read_csv(os.path.join(EMB_DIR, "titanet_metadata_agevoxceleb_valid.csv"))
print(f"   Метаданные загружены, {len(df_meta)} записей")

print("\n3. Загрузка модели TLM...")
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)
print(f"   Модель загружена, тип: {type(model)}")

print("\n4. Выполнение предсказаний...")
predictions = model.predict(all_emb)
print(f"   Предсказания получены, форма: {predictions.shape}")

print("\n5. Сохранение результатов...")
df_meta['age_pred'] = predictions

# Сохраняем результат
output_csv = os.path.join(OUTPUT_DIR, "agevox_titanet_metadata_with_predictions.csv")
df_meta.to_csv(output_csv, index=False)
print(f"   Результат сохранён в {output_csv}")

np.save(os.path.join(OUTPUT_DIR, "age_predictions_agevoxceleb.npy"), predictions)
print(f"   Предсказания сохранены в age_predictions_agevoxceleb.npy")

print("\n6. Статистика предсказанных возрастов:")
print(f"   Min: {predictions.min():.1f}")
print(f"   Max: {predictions.max():.1f}")
print(f"   Mean: {predictions.mean():.1f}")
print(f"   Std: {predictions.std():.1f}")

print("\n7. Первые 10 предсказаний:")
print(predictions[:10])