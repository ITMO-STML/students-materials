import os
import torch
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
import ssl
import librosa
import io
import soundfile as sf
import pyarrow.parquet as pq
from pathlib import Path

import nemo.collections.asr as nemo_asr

ssl._create_default_https_context = ssl._create_unverified_context
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Пути
DATASET_DIR = "/home/ext-ivanova-mk@ad.speechpro.com/ITMO/datasets--evan0617--seniortalk/snapshots/d8f71863fff5d3128f806ca9025c653dd3dac397/sentence_data/"
OUTPUT_DIR = "/home/ext-ivanova-mk@ad.speechpro.com/test_dir/tlm/feature_extraction/seniortalk_titanet_embeddings"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_seniortalk_audio(audio_bytes, target_sr=16000, max_samples=240000, min_samples=48000):
    try:
        audio_file = io.BytesIO(audio_bytes)
        audio, sr = sf.read(audio_file)
        
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        elif len(audio) < min_samples:
            return None
        
        audio = audio / (np.abs(audio).max() + 1e-8)
        return torch.FloatTensor(audio).unsqueeze(0)
    
    except Exception as e:
        logging.error(f"Ошибка аудио: {e}")
        return None

def load_seniortalk_from_parquet(dataset_dir):
    """
    Загружает данные SeniorTalk из parquet файлов
    """
    audio_data = []
    metadata_list = []
    
    parquet_files = list(Path(dataset_dir).glob("*.parquet"))
    
    if not parquet_files:
        logging.error(f"Не найдены parquet файлы в {dataset_dir}")
        return None, None
    
    logging.info(f"Найдено {len(parquet_files)} parquet файлов")
    
    # Только test файлы
    test_files = [f for f in parquet_files if 'test' in str(f)]
    if not test_files:
        test_files = parquet_files
    
    logging.info(f"Обрабатываем {len(test_files)} файлов")
    
    for parquet_file in test_files:
        try:
            logging.info(f"Чтение {parquet_file}...")
            table = pq.read_table(parquet_file)
            df = table.to_pandas()
            
            for idx, row in df.iterrows():
                if 'path' in row and isinstance(row['path'], dict):
                    if 'bytes' in row['path']:
                        audio_bytes = row['path']['bytes']
                        filename = row['path'].get('path', f"sample_{idx}.wav")
                        
                        speaker_id = None
                        import re
                        match = re.search(r'Elderly(\d+)S', filename)
                        if match:
                            speaker_id = match.group(1)
                        
                        audio_data.append({
                            'audio_bytes': audio_bytes,
                            'filename': filename,
                            'speaker_id': speaker_id
                        })
                        
                        metadata_list.append({
                            'file_name': filename,
                            'speaker_id': speaker_id,
                            'idx': len(audio_data) - 1
                        })
                        
                        if len(audio_data) % 1000 == 0:
                            logging.info(f"Загружено {len(audio_data)} записей")
                            
        except Exception as e:
            logging.error(f"Ошибка чтения {parquet_file}: {e}")
            continue
    
    logging.info(f"Всего загружено {len(audio_data)} аудиозаписей")
    return audio_data, metadata_list

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
    
    logging.info(f"Загрузка SeniorTalk из {DATASET_DIR}...")
    audio_data, metadata_list = load_seniortalk_from_parquet(DATASET_DIR)
    
    if audio_data is None or len(audio_data) == 0:
        logging.error("Не удалось загрузить данные")
        return
    
    # Извлекаем эмбеддинги
    embedding_dim = None
    processed_count = 0
    error_count = 0
    
    error_log = os.path.join(OUTPUT_DIR, "errors.log")
    valid_metadata = []
    
    with torch.no_grad():
        for i, item in enumerate(tqdm(audio_data, desc="Извлечение эмбеддингов")):
            try:
                audio_bytes = item['audio_bytes']
                filename = item['filename']
                speaker_id = item['speaker_id']
                
                audio_tensor = load_seniortalk_audio(audio_bytes)
                if audio_tensor is None:
                    error_count += 1
                    with open(error_log, 'a') as f_err:
                        f_err.write(f"LOAD_ERROR: {filename}\n")
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
                safe_filename = filename.replace('.wav', '').replace('/', '_')
                emb_path = os.path.join(OUTPUT_DIR, f"{safe_filename}.npy")
                np.save(emb_path, embeddings.cpu().numpy())
                
                valid_metadata.append({
                    'file_name': filename,
                    'speaker_id': speaker_id,
                    'embedding_path': emb_path,
                    'embedding_dim': embedding_dim
                })
                
                processed_count += 1
                
                if (i + 1) % 500 == 0:
                    logging.info(f"Обработано {processed_count}/{len(audio_data)} файлов")
                    
            except Exception as e:
                error_count += 1
                with open(error_log, 'a') as f_err:
                    f_err.write(f"PROCESS_ERROR: {filename} - {str(e)}\n")
                continue
    
    # Сохраняем метаданные
    df_metadata = pd.DataFrame(valid_metadata)
    df_metadata.to_csv(os.path.join(OUTPUT_DIR, "seniortalk_titanet_metadata.csv"), index=False)
    
    # Сборка единого массива
    if valid_metadata:
        all_embeddings = [np.load(item['embedding_path']) for item in valid_metadata]
        all_embeddings = np.array(all_embeddings)
        np.save(os.path.join(OUTPUT_DIR, "all_embeddings_seniortalk.npy"), all_embeddings)
        logging.info(f"Сохранен массив shape: {all_embeddings.shape}")
    
    print("\nЗавершено")
    print(f"Всего записей: {len(audio_data)}")
    print(f"Успешно обработано: {processed_count}")
    print(f"Ошибок: {error_count}")
    print(f"Размерность эмбеддинга: {embedding_dim}")
    print(f"Эмбеддинги сохранены в: {OUTPUT_DIR}")
    
    if valid_metadata:
        print("\nПример успешно обработанного файла:")
        sample = valid_metadata[0]
        print(f"  файл: {sample['file_name']}")
        print(f"  speaker_id: {sample['speaker_id']}")
        print(f"  эмбеддинг: {sample['embedding_path']}")
        sample_emb = np.load(sample['embedding_path'])
        print(f"  форма эмбеддинга: {sample_emb.shape}")

if __name__ == "__main__":
    main()