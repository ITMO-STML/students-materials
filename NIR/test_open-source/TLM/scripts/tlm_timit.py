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

EMB_DIR = "/home/ext-ivanova-mk@ad.speechpro.com/test_dir/tlm/feature_extraction/timit_titanet_embeddings"
MODEL_PATH = "/home/ext-ivanova-mk@ad.speechpro.com/test_dir/tlm/models/tlm_age_mixup_3.pkl"
OUTPUT_DIR = EMB_DIR

print("\n1. Загрузка эмбеддингов...")
all_emb = np.load(os.path.join(EMB_DIR, "all_embeddings.npy"))
all_emb = all_emb.squeeze(1)

print("\n2. Загрузка модели TLM...")
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)
print(f"   Модель загружена, тип: {type(model)}")

print("\n3. Выполнение предсказаний...")
predictions = model.predict(all_emb)

# Загружаем метаданные
print("\n4. Сохранение результатов...")
df_meta = pd.read_csv(os.path.join(EMB_DIR, "timit_titanet_metadata.csv"))
df_meta['age_pred'] = predictions

# Сохраняем результат
output_csv = os.path.join(OUTPUT_DIR, "timit_titanet_metadata_with_predictions.csv")
df_meta.to_csv(output_csv, index=False)
print(f"   Результат сохранён в {output_csv}")

# Сохраняем предсказания в отдельный файл
np.save(os.path.join(OUTPUT_DIR, "age_predictions.npy"), predictions)
print(f"   Предсказания сохранены в age_predictions.npy")

print("\n5. Статистика предсказанных возрастов:")
print(f"   Min: {predictions.min():.1f}")
print(f"   Max: {predictions.max():.1f}")
print(f"   Mean: {predictions.mean():.1f}")
print(f"   Std: {predictions.std():.1f}")

print("\n6. Первые 10 предсказаний:")
print(predictions[:10])