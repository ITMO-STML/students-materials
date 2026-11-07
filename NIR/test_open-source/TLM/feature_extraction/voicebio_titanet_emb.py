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

DATASET_PATH = "/mnt/storage/work_dir/databases/voice_biometrics_age/voice_biometrics_age"
EXCEL_PATH = "/mnt/storage/work_dir/databases/voice_biometrics_age/metadata.xlsx"
OUTPUT_DIR = "/home/ext-ivanova-mk@ad.speechpro.com/test_dir/tlm/feature_extraction/voicebio_titanet_embeddings"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_audio(file_path, target_sr=16000, max_samples=240000, min_samples=48000):

    try:
        audio, sr = sf.read(file_path)
        
        # Конвертируем в моно если стерео
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Ресемплинг
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        
        # Ограничение длины (15 секунд)
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        elif len(audio) < min_samples:  # Минимум 3 секунды
            return None
        
        # Нормализация
        audio = audio / (np.abs(audio).max() + 1e-8)
        
        return torch.FloatTensor(audio).unsqueeze(0)
    
    except Exception as e:
        logging.error(f"Ошибка загрузки {file_path}: {e}")
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
    
    logging.info(f"Загрузка метаданных из {EXCEL_PATH}...")
    df_persons = pd.read_excel(EXCEL_PATH, engine='openpyxl')
    df_persons['date of birth'] = pd.to_datetime(df_persons['date of birth'], errors='coerce')
    
    id_to_birth_year = {}
    id_to_gender = {}
    for _, row in df_persons.iterrows():
        person_id = row['id']
        if pd.notna(row['date of birth']):
            id_to_birth_year[person_id] = row['date of birth'].year
            id_to_gender[person_id] = row.get('gender (male or female)', 'unknown')
    
    logging.info(f"Загружено {len(id_to_birth_year)} персон с известной датой рождения")
    
    audio_files = []
    metadata_list = []
    
    for person_dir in Path(DATASET_PATH).iterdir():
        if not person_dir.is_dir() or not person_dir.name.startswith('id'):
            continue
        
        person_id = person_dir.name
        
        if person_id not in id_to_birth_year:
            logging.warning(f"Нет даты рождения для {person_id}, пропускаем")
            continue
        
        birth_year = id_to_birth_year[person_id]
        gender = id_to_gender.get(person_id, 'unknown')
        
        for year_dir in person_dir.iterdir():
            if not year_dir.is_dir():
                continue
            
            try:
                record_year = int(year_dir.name)
                age = record_year - birth_year
                
                if age < 0 or age > 100:
                    continue
                
                wav_files = list(year_dir.glob('*.wav'))
                
                for audio_path in wav_files:
                    audio_files.append(str(audio_path))
                    metadata_list.append({
                        'person_id': person_id,
                        'birth_year': birth_year,
                        'record_year': record_year,
                        'age': age,
                        'gender': gender,
                        'file_path': str(audio_path)
                    })
                        
            except ValueError:
                continue
    
    logging.info(f"Найдено {len(audio_files)} аудиофайлов")
    
    embedding_dim = None
    processed_count = 0
    error_count = 0
    
    error_log = os.path.join(OUTPUT_DIR, "errors.log")
    valid_metadata = []
    
    with torch.no_grad():
        for i, (audio_path, metadata) in enumerate(tqdm(zip(audio_files, metadata_list), 
                                                         desc="Извлечение эмбеддингов", 
                                                         total=len(audio_files))):
            try:
                audio_tensor = load_audio(audio_path)
                if audio_tensor is None:
                    error_count += 1
                    with open(error_log, 'a') as f_err:
                        f_err.write(f"LOAD_ERROR: {audio_path}\n")
                    continue
                
                audio_tensor = audio_tensor.to(device)
                length = torch.tensor([audio_tensor.shape[1]], device=device)
                
                output = speaker_model.forward(input_signal=audio_tensor, input_signal_length=length)
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
                # Имя файла: id00001_2020_001.wav -> id00001_2020_001.npy
                file_name = os.path.basename(audio_path).replace('.wav', '')
                emb_filename = f"{metadata['person_id']}_{metadata['record_year']}_{file_name}.npy"
                emb_path = os.path.join(OUTPUT_DIR, emb_filename)
                np.save(emb_path, embeddings.cpu().numpy())
                
                metadata['embedding_path'] = emb_path
                metadata['embedding_dim'] = embedding_dim
                
                valid_metadata.append(metadata)
                processed_count += 1
                
                if (i + 1) % 500 == 0:
                    logging.info(f"Обработано {processed_count}/{len(audio_files)} файлов")
                    
            except Exception as e:
                error_count += 1
                with open(error_log, 'a') as f_err:
                    f_err.write(f"PROCESS_ERROR: {audio_path} - {str(e)}\n")
                continue
    
    # Сохраняем метаданные
    df_metadata = pd.DataFrame(valid_metadata)
    df_metadata.to_csv(os.path.join(OUTPUT_DIR, "voicebio_titanet_metadata.csv"), index=False)
    
    logging.info("Сборка единого массива эмбеддингов...")
    all_embeddings = []
    
    for item in tqdm(valid_metadata, desc="Сборка единого массива"):
        emb = np.load(item['embedding_path'])
        all_embeddings.append(emb)
    
    if all_embeddings:
        all_embeddings = np.array(all_embeddings)
        np.save(os.path.join(OUTPUT_DIR, "all_embeddings_voicebio.npy"), all_embeddings)
        logging.info(f"Сохранен массив эмбеддингов shape: {all_embeddings.shape}")
    
    # Итоговая статистика
    print("\nЗавершено")
    print(f"Всего людей с датой рождения: {len(id_to_birth_year)}")
    print(f"Найдено людей в датасете: {df_metadata['person_id'].nunique() if len(valid_metadata) > 0 else 0}")
    print(f"Всего аудиофайлов: {len(audio_files)}")
    print(f"Успешно обработано: {processed_count}")
    print(f"Ошибок: {error_count}")
    print(f"Размерность эмбеддинга: {embedding_dim}")
    print(f"Эмбеддинги сохранены в: {OUTPUT_DIR}")
    print(f"Метаданные: {os.path.join(OUTPUT_DIR, 'voicebio_titanet_metadata.csv')}")
    if error_count > 0:
        print(f"Лог ошибок: {error_log}")
    
    # Пример
    if valid_metadata:
        print("\nПример успешно обработанного файла:")
        sample = valid_metadata[0]
        print(f"  person_id: {sample['person_id']}")
        print(f"  возраст: {sample['age']} лет")
        print(f"  год записи: {sample['record_year']}")
        print(f"  пол: {sample['gender']}")
        print(f"  файл: {sample['file_path']}")
        print(f"  эмбеддинг: {sample['embedding_path']}")
        sample_emb = np.load(sample['embedding_path'])
        print(f"  форма эмбеддинга: {sample_emb.shape}")

if __name__ == "__main__":
    main()