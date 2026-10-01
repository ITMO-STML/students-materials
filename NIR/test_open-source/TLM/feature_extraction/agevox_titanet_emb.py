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

VOXCELEB2_PATH = "/mnt/storage/work_dir/databases/voxceleb2/" 
METADATA_PATH = "/home/ext-ivanova-mk@ad.speechpro.com/test_dir/vox-profile-release/agevoxceleb/utt2age.test"
OUTPUT_DIR = "/home/ext-ivanova-mk@ad.speechpro.com/test_dir/tlm/feature_extraction/agevox_titanet_embeddings"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def find_agevoxceleb_wav_files(metadata_path, base_audio_path):

    df_age = pd.read_csv(metadata_path, sep='\s+', header=None, names=['utterance_id', 'age'])
    utterance_ids = df_age['utterance_id'].tolist()
    
    wav_files = []
    metadata_list = []
    
    logging.info(f"Загружено {len(utterance_ids)} записей из Age-VoxCeleb")
    logging.info("Поиск аудиофайлов...")
    
    for utt_id in utterance_ids:
        parts = utt_id.split('/')
        if len(parts) >= 3:
            speaker, video, segment = parts[0], parts[1], parts[2]
            
            wav_path = os.path.join(base_audio_path, speaker, video, f"{segment}.wav")
            
            if os.path.exists(wav_path):
                wav_files.append(wav_path)
                metadata_list.append({
                    'utterance_id': utt_id,
                    'speaker_id': speaker,
                    'video_id': video,
                    'segment': segment,
                    'age': df_age[df_age['utterance_id'] == utt_id]['age'].iloc[0],
                    'wav_path': wav_path
                })
            else:
                m4a_path = os.path.join(base_audio_path, speaker, video, f"{segment}.m4a")
                if os.path.exists(m4a_path):
                    wav_files.append(m4a_path)
                    metadata_list.append({
                        'utterance_id': utt_id,
                        'speaker_id': speaker,
                        'video_id': video,
                        'segment': segment,
                        'age': df_age[df_age['utterance_id'] == utt_id]['age'].iloc[0],
                        'wav_path': m4a_path
                    })
    
    logging.info(f"Найдено {len(wav_files)} аудиофайлов из {len(utterance_ids)} записей")
    
    return wav_files, metadata_list

def load_audio(file_path, target_sr=16000, max_samples=240000):

    try:
        audio, sr = sf.read(file_path)
        
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        
        # Ограничение длины (15 секунд)
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        elif len(audio) < 48000:  # Минимум 3 секунды
            logging.warning(f"Аудио слишком короткое ({len(audio)/target_sr:.2f} сек): {file_path}")
            return None
        
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
    
    wav_files, metadata_list = find_agevoxceleb_wav_files(METADATA_PATH, VOXCELEB2_PATH)
    
    if len(wav_files) == 0:
        logging.error("Аудиофайлы не найдены! Проверьте пути.")
        return
    
    embedding_dim = None
    processed_count = 0
    error_count = 0
    
    error_log = os.path.join(OUTPUT_DIR, "errors.log")
    
    with torch.no_grad():
        for i, wav_path in enumerate(tqdm(wav_files, desc="Извлечение эмбеддингов")):
            try:
                audio = load_audio(wav_path, target_sr=16000, max_samples=240000)
                if audio is None:
                    error_count += 1
                    with open(error_log, 'a') as f_err:
                        f_err.write(f"LOAD_ERROR: {wav_path}\n")
                    continue
                
                audio = audio.to(device)
                audio_batch = audio.unsqueeze(0)
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
                emb_filename = f"{metadata['speaker_id']}_{metadata['video_id']}_{metadata['segment']}.npy"
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
    df_metadata.to_csv(os.path.join(OUTPUT_DIR, "agevox_titanet_metadata.csv"), index=False)
    
    # Сборка единого массива
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
        np.save(os.path.join(OUTPUT_DIR, "all_embeddings_agevoxceleb.npy"), all_embeddings)
        
        # Сохраняем метаданные для успешно обработанных
        df_valid = pd.DataFrame(valid_metadata)
        df_valid.to_csv(os.path.join(OUTPUT_DIR, "titanet_metadata_agevoxceleb_valid.csv"), index=False)
        
        logging.info(f"Сохранен массив эмбеддингов shape: {all_embeddings.shape}")
    
    # Итоговая статистика
    print("\nЗавершено")
    print(f"Всего записей в метаданных: {len(metadata_list)}")
    print(f"Успешно обработано: {processed_count}")
    print(f"Ошибок: {error_count}")
    print(f"Размерность эмбеддинга: {embedding_dim}")
    print(f"Эмбеддинги сохранены в: {OUTPUT_DIR}")
    print(f"Метаданные: {os.path.join(OUTPUT_DIR, 'agevox_titanet_metadata.csv')}")
    if error_count > 0:
        print(f"Лог ошибок: {error_log}")
    
    if valid_metadata:
        print("\nПример успешно обработанного файла:")
        sample = valid_metadata[0]
        print(f"  utterance_id: {sample['utterance_id']}")
        print(f"  возраст: {sample['age']}")
        print(f"  эмбеддинг: {sample['embedding_path']}")
        sample_emb = np.load(sample['embedding_path'])
        print(f"  форма эмбеддинга: {sample_emb.shape}")

if __name__ == "__main__":
    main()