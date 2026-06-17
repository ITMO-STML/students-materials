import os
import torch
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
import ssl
import soundfile as sf
import librosa
from pathlib import Path

import nemo.collections.asr as nemo_asr

ssl._create_default_https_context = ssl._create_unverified_context
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

NNCES_PATH = "/home/ext-ivanova-mk@ad.speechpro.com/ITMO/nnces_split/test"
OUTPUT_DIR = "/home/ext-ivanova-mk@ad.speechpro.com/test_dir/tlm/feature_extraction/nnces_titanet_embeddings"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def find_nnces_wav_files(base_path):

    wav_files = []
    metadata_list = []
    
    logging.info(f"Поиск WAV файлов в {base_path}...")
    
    for wav_path in Path(base_path).rglob('*.wav'):
        wav_files.append(str(wav_path))
        
        rel_path = os.path.relpath(wav_path, base_path)
        parts = rel_path.split(os.sep)
        
        age = None
        for part in parts:
            if part.endswith('Y') and part[:-1].isdigit():
                age = int(part[:-1])
                break
        
        gender = None
        speaker_age = None
        for part in parts:
            part_lower = part.lower()
            if 'boy' in part_lower:
                gender = 'M'
                numbers = ''.join([c for c in part if c.isdigit()])
                if numbers:
                    speaker_age = int(numbers)
            elif 'girl' in part_lower:
                gender = 'F'
                numbers = ''.join([c for c in part if c.isdigit()])
                if numbers:
                    speaker_age = int(numbers)
        
        speaker_id = None
        for part in parts:
            if len(part) >= 2 and part[0] in ['M', 'F'] and part[1:].isdigit():
                speaker_id = part
                break
        
        metadata_list.append({
            'file_path': wav_path,
            'age': age,
            'gender': gender,
            'speaker_age': speaker_age,
            'speaker_id': speaker_id,
            'rel_path': rel_path
        })
    
    logging.info(f"Найдено {len(wav_files)} WAV файлов")
    
    return wav_files, metadata_list

def load_audio(file_path, target_sr=16000, max_samples=240000, min_samples=48000):
    """
    Загружает аудио, возвращает 1D тензор torch
    Ограничивает длину до max_samples (15 секунд при 16kHz)
    """
    try:
        audio, sr = sf.read(file_path)
        
        # Конвертируем в моно если стерео
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Ресемплинг если нужно
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        
        # Ограничение длины (15 секунд)
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        elif len(audio) < min_samples:  # Минимум 3 секунды
            logging.warning(f"Аудио слишком короткое ({len(audio)/target_sr:.2f} сек): {file_path}")
            return None
        
        # Нормализация
        audio = audio / (np.abs(audio).max() + 1e-8)
        
        return torch.from_numpy(audio).float()
    except Exception as e:
        logging.warning(f"Ошибка загрузки {file_path}: {e}")
        return None

def main():
    model_path = "/home/ext-ivanova-mk@ad.speechpro.com/.cache/huggingface/hub/models--nvidia--speakerverification_en_titanet_large/snapshots/0dc382f40121a5fbd34db10a2bb04d826c2be6a8/speakerverification_en_titanet_large.nemo"
    
    try:
        speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.restore_from(model_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        speaker_model = speaker_model.to(device)
        speaker_model.eval()
        logging.info(f"Модель TitaNet загружена на устройство: {device}")
    except Exception as e:
        logging.error(f"Ошибка загрузки модели: {e}")
        return
    
    # Находим все WAV файлы
    wav_files, metadata_list = find_nnces_wav_files(NNCES_PATH)
    
    if len(wav_files) == 0:
        logging.error("Аудиофайлы не найдены! Проверьте путь.")
        return
    
    embedding_dim = None
    processed_count = 0
    error_count = 0
    
    error_log = os.path.join(OUTPUT_DIR, "errors.log")
    
    with torch.no_grad():
        for i, wav_path in enumerate(tqdm(wav_files, desc="Извлечение эмбеддингов")):
            try:
                audio = load_audio(wav_path, target_sr=16000, max_samples=240000, min_samples=48000)
                if audio is None:
                    error_count += 1
                    with open(error_log, 'a') as f_err:
                        f_err.write(f"LOAD_ERROR: {wav_path}\n")
                    continue
                
                audio = audio.to(device)
                audio_batch = audio.unsqueeze(0)  # [1, samples]
                length = torch.tensor([audio_batch.shape[1]], device=device)
                
                output = speaker_model.forward(input_signal=audio_batch, input_signal_length=length)
                if isinstance(output, tuple):
                    embeddings = output[1]
                else:
                    embeddings = output
                
                if embeddings.dim() == 3:
                    embeddings = embeddings.mean(dim=1)
                
                if embedding_dim is None:
                    embedding_dim = embeddings.shape[1]
                    logging.info(f"Размерность эмбеддинга: {embedding_dim}")
                
                # Сохраняем эмбеддинг
                metadata = metadata_list[i]
                # Создаем имя файла: speaker_id_age_gender_original_name.npy
                file_name = os.path.basename(wav_path).replace('.wav', '')
                if metadata['speaker_id']:
                    emb_filename = f"{metadata['speaker_id']}_{metadata['age']}Y_{file_name}.npy"
                else:
                    emb_filename = f"{metadata['age']}Y_{file_name}.npy"
                
                emb_path = os.path.join(OUTPUT_DIR, emb_filename)
                np.save(emb_path, embeddings.cpu().numpy())
                
                metadata['embedding_path'] = emb_path
                metadata['embedding_dim'] = embedding_dim
                
                processed_count += 1
                
                if (i + 1) % 1000 == 0:
                    logging.info(f"Обработано {processed_count}/{len(wav_files)} файлов")
                    
            except Exception as e:
                error_count += 1
                with open(error_log, 'a') as f_err:
                    f_err.write(f"PROCESS_ERROR: {wav_path} - {str(e)}\n")
                continue
    
    df_metadata = pd.DataFrame(metadata_list)
    df_metadata.to_csv(os.path.join(OUTPUT_DIR, "nnces_titanet_metadata.csv"), index=False)
    
    logging.info("Сборка единого массива эмбеддингов...")
    all_embeddings = []
    valid_metadata = []
    
    for item in tqdm(metadata_list, desc="Сборка единого массива"):
        if 'embedding_path' in item and os.path.exists(item['embedding_path']):
            emb = np.load(item['embedding_path'])
            all_embeddings.append(emb)
            valid_metadata.append(item)
    
    if all_embeddings:
        all_embeddings = np.array(all_embeddings)
        np.save(os.path.join(OUTPUT_DIR, "all_embeddings_nnces.npy"), all_embeddings)
        
        df_valid = pd.DataFrame(valid_metadata)
        df_valid.to_csv(os.path.join(OUTPUT_DIR, "nnces_titanet_metadata_valid.csv"), index=False)
        
        logging.info(f"Сохранен массив эмбеддингов shape: {all_embeddings.shape}")
    
    print("Извлечение завершено")
    print(f"Всего записей: {len(metadata_list)}")
    print(f"Успешно обработано: {processed_count}")
    print(f"Ошибок: {error_count}")
    print(f"Размерность эмбеддинга: {embedding_dim}")
    print(f"Эмбеддинги сохранены в: {OUTPUT_DIR}")
    print(f"Метаданные: {os.path.join(OUTPUT_DIR, 'nnces_titanet_metadata.csv')}")
    if error_count > 0:
        print(f"Лог ошибок: {error_log}")
    
    if valid_metadata:
        print("\nПример обработанного файла:")
        sample = valid_metadata[0]
        print(f"  файл: {sample['file_path']}")
        print(f"  возраст: {sample['age']}")
        print(f"  пол: {sample['gender']}")
        print(f"  эмбеддинг: {sample['embedding_path']}")
        sample_emb = np.load(sample['embedding_path'])
        print(f"  форма эмбеддинга: {sample_emb.shape}")

if __name__ == "__main__":
    main()