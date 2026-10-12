import os
import torch
import torchaudio
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from tqdm import tqdm
from transformers import WavLMModel
import librosa
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


# Настройка логирования
logging.basicConfig(level=logging.INFO)

# Пути
TIMIT_PATH = "./darpa-timit-acousticphonetic-continuous-speech/versions/6/"
OUTPUT_DIR = "./feature_extraction/wavlm_emb/wavlm_original_embeddings"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Функции для поиска файлов TIMIT
def find_original_wav_files(timit_base_path):
    """Находит оригинальные .WAV файлы"""
    wav_files = []
    def _search(path):
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path):
                _search(item_path)
            elif item.endswith('.WAV'):
                wav_files.append(item_path)
    _search(timit_base_path)
    return sorted(wav_files)

def get_speaker_id_from_path(file_path):
    """Извлекает speaker_id из пути (отбрасывая первую букву пола)"""
    parts = file_path.split(os.sep)
    for part in parts:
        if len(part) == 5 and part[0] in ['F', 'M'] and part[1:].isalnum():
            return part[1:]  # FAKS0 -> AKS0
    return None

def load_audio_torch(file_path, target_sr=16000):
    """Загружает аудио через torchaudio"""
    try:
        audio, sr = torchaudio.load(file_path)
        if sr != target_sr:
            audio = torchaudio.functional.resample(audio, sr, target_sr)
        # Нормализация
        audio = audio / (audio.abs().max() + 1e-8)
        return audio  # [1, samples]
    except Exception as e:
        logging.warning(f"Ошибка загрузки {file_path}: {e}")
        return None

# Загружаем оригинальную WavLM модель
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Использую устройство: {device}")

model = WavLMModel.from_pretrained("microsoft/wavlm-large")
model = model.to(device)
model.eval()
logging.info("Модель WavLM загружена")

# Находим все WAV файлы в TIMIT
wav_files = find_original_wav_files(TIMIT_PATH)
logging.info(f"Найдено {len(wav_files)} файлов")

# Извлекаем эмбеддинги
all_embeddings = []
metadata = []

with torch.no_grad():
    for wav_path in tqdm(wav_files, desc="Извлечение эмбеддингов"):
        # Получаем speaker_id
        speaker_id = get_speaker_id_from_path(wav_path)
        if speaker_id is None:
            continue
        
        # Загружаем аудио
        audio = load_audio_torch(wav_path, target_sr=16000)
        if audio is None:
            continue
        
        # Перемещаем на устройство
        audio = audio.to(device)
        
        # Извлекаем эмбеддинги
        outputs = model(audio)  # [1, time, 1024]
        
        # Берем среднее по времени (это стандартный подход)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # [1, 1024]
        
        # Сохраняем эмбеддинг
        base_name = os.path.basename(wav_path).replace('.WAV', '')
        speaker_full = os.path.basename(os.path.dirname(wav_path))
        
        emb_path = f"{OUTPUT_DIR}/{speaker_full}_{base_name}.npy"
        np.save(emb_path, embeddings.cpu().numpy())
        
        # Сохраняем метаданные
        metadata.append({
            'speaker_id': speaker_id,
            'speaker_full': speaker_full,
            'gender': 'M' if speaker_full[0] == 'M' else 'F',
            'file_name': base_name,
            'file_path': wav_path,
            'embedding_path': emb_path
        })
        
        if len(metadata) % 100 == 0:
            logging.info(f"Обработано {len(metadata)} файлов")

# Сохраняем метаданные
df = pd.DataFrame(metadata)
df.to_csv(f"{OUTPUT_DIR}/wavlm_metadata.csv", index=False)

logging.info(f"Сохранено {len(metadata)} эмбеддингов")
logging.info(f"Размерность эмбеддинга: {embeddings.shape[1]}")