import os
import torch
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
import ssl
import soundfile as sf
import librosa

import nemo.collections.asr as nemo_asr

ssl._create_default_https_context = ssl._create_unverified_context
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

TIMIT_PATH = "/home/ext-ivanova-mk@ad.speechpro.com/ITMO/darpa-timit-acousticphonetic-continuous-speech/versions/6/"
OUTPUT_DIR = "/home/ext-ivanova-mk@ad.speechpro.com/test_dir/tlm/feature_extraction/titanet_embeddings"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def find_original_wav_files(timit_base_path):
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
    parts = file_path.split(os.sep)
    for part in parts:
        if len(part) == 5 and part[0] in ['F', 'M'] and part[1:].isalnum():
            return {
                'full_speaker': part,
                'speaker_id': part[1:],
                'gender': 'M' if part[0] == 'M' else 'F'
            }
    return None

def load_audio(file_path, target_sr=16000):
    try:
        audio, sr = sf.read(file_path)
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        audio = audio / (np.abs(audio).max() + 1e-8)
        audio = torch.from_numpy(audio).float()  # [samples]
        return audio
    except Exception as e:
        logging.warning(f"Ошибка загрузки {file_path}: {e}")
        return None

model_path = "/home/ext-ivanova-mk@ad.speechpro.com/.cache/huggingface/hub/models--nvidia--speakerverification_en_titanet_large/snapshots/0dc382f40121a5fbd34db10a2bb04d826c2be6a8/speakerverification_en_titanet_large.nemo"

speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.restore_from(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
speaker_model = speaker_model.to(device)
speaker_model.eval()
logging.info("Модель TitaNet загружена из кэша")

wav_files = find_original_wav_files(TIMIT_PATH)
logging.info(f"Найдено {len(wav_files)} файлов")

metadata = []
embedding_dim = None

with torch.no_grad():
    for wav_path in tqdm(wav_files, desc="Извлечение эмбеддингов"):
        speaker_info = get_speaker_id_from_path(wav_path)
        if speaker_info is None:
            continue
        
        audio = load_audio(wav_path, target_sr=16000)
        if audio is None:
            continue
        
        audio = audio.to(device)
        audio_batch = audio.unsqueeze(0)
        length = torch.tensor([audio_batch.shape[1]], device=device)
        
        output = speaker_model.forward(input_signal=audio_batch, input_signal_length=length)
        if isinstance(output, tuple):
            embeddings = output[1]
        else:
            embeddings = output
        
        # Если результат [batch, time, dim], усредняем по времени
        if embeddings.dim() == 3:
            embeddings = embeddings.mean(dim=1)
        
        base_name = os.path.basename(wav_path).replace('.WAV', '')
        speaker_full = speaker_info['full_speaker']
        emb_path = os.path.join(OUTPUT_DIR, f"{speaker_full}_{base_name}.npy")
        np.save(emb_path, embeddings.cpu().numpy())
        
        metadata.append({
            'speaker_full': speaker_full,
            'speaker_id': speaker_info['speaker_id'],
            'gender': speaker_info['gender'],
            'file_name': base_name,
            'file_path': wav_path,
            'embedding_path': emb_path
        })
        
        if embedding_dim is None:
            embedding_dim = embeddings.shape[1]
            logging.info(f"Размерность эмбеддинга: {embedding_dim}")
        
        if len(metadata) % 100 == 0:
            logging.info(f"Обработано {len(metadata)} файлов")

df = pd.DataFrame(metadata)
df.to_csv(os.path.join(OUTPUT_DIR, "titanet_metadata.csv"), index=False)

if metadata:
    logging.info(f"Готово! Сохранено {len(metadata)} эмбеддингов")
    logging.info(f"Размерность эмбеддинга: {embedding_dim}")
    logging.info(f"Путь: {OUTPUT_DIR}")
    sample_emb = np.load(metadata[0]['embedding_path'])
    logging.info(f"Пример shape: {sample_emb.shape}")
else:
    logging.error("Не удалось обработать ни одного файла!")

# Сборка единого массива
all_embeddings = []
for item in tqdm(metadata, desc="Сборка единого массива"):
    emb = np.load(item['embedding_path'])
    all_embeddings.append(emb)

if all_embeddings:
    all_embeddings = np.array(all_embeddings)
    np.save(os.path.join(OUTPUT_DIR, "all_embeddings.npy"), all_embeddings)
    logging.info(f"Сохранен массив shape: {all_embeddings.shape}")

print("\nИзвлечение эмбеддингов TitaNet завершено!")
print(f"Эмбеддинги сохранены в: {OUTPUT_DIR}")