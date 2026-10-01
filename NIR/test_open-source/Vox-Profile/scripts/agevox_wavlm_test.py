import torch
import sys, os
import torch.nn.functional as F
import argparse, logging
from pathlib import Path
import numpy as np
import ssl
import pandas as pd

ssl._create_default_https_context = ssl._create_unverified_context

sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[1])))
sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[1]), 'model', 'age_sex'))

from model.wavlm_demographics import WavLMWrapper

import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-3s ==> %(message)s', 
    level=logging.INFO, 
    datefmt='%Y-%m-%d %H:%M:%S'
)

os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1"

def load_voxceleb_audio(file_path, target_sr=16000, max_samples=240000):
    """Загружает аудио из VoxCeleb2 (WAV файлы)"""
    try:
        import librosa
        
        # Загружаем с исходной частотой
        audio, sr = librosa.load(file_path, sr=None, mono=True)
        
        # РЕСЕМПЛИНГ если нужно
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        
        # Ограничение длины (15 секунд)
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        elif len(audio) < 48000:  # Минимум 3 секунды
            return None
                
        # Нормализация
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        
        return torch.FloatTensor(audio).unsqueeze(0)
    
    except Exception as e:
        logging.error(f"Ошибка загрузки {file_path}: {e}")
        return None

def find_voxceleb_wav_files(base_path, utterance_ids):
    """Находит WAV файлы по utterance_id в VoxCeleb2"""
    wav_files = []
    
    for utt_id in utterance_ids:
        # Пример utterance_id: "id00042/34f5sBShxSo/00001"
        parts = utt_id.split('/')
        if len(parts) >= 3:
            speaker, video, segment = parts[0], parts[1], parts[2]
            
            # Пробуем WAV файл
            wav_path = os.path.join(base_path, speaker, video, f"{segment}.wav")
            
            if os.path.exists(wav_path):
                wav_files.append(wav_path)
            else:
                # Пробуем M4A если нет WAV
                m4a_path = os.path.join(base_path, speaker, video, f"{segment}.m4a")
                if os.path.exists(m4a_path):
                    wav_files.append(m4a_path)
    
    return wav_files

if __name__ == '__main__':
    # Настройки для Age-VoxCeleb
    voxceleb2_base_path = "./voxceleb2/"
    metadata_path = "./agevoxceleb/utt2age.test"
    output_file = "agevoxceleb_age_predictions.txt"
    
    # Загружаем метаданные
    logging.info(f"Загрузка метаданных из {metadata_path}...")
    df_age = pd.read_csv(metadata_path, sep='\s+', header=None, names=['utterance_id', 'age'])
    
    # Берем ВСЕ тестовые данные
    utterance_ids = df_age['utterance_id'].tolist()
    logging.info(f"Загружено {len(utterance_ids)} записей из Age-VoxCeleb")
    
    # Поиск файлов
    logging.info(f"Поиск аудиофайлов в {voxceleb2_base_path}...")
    wav_files = find_voxceleb_wav_files(voxceleb2_base_path, utterance_ids)
    
    logging.info(f"Найдено {len(wav_files)} аудиофайлов из {len(utterance_ids)} записей")
    
    if len(wav_files) == 0:
        logging.error("Файлы не найдены! Выход.")
        exit(1)
    
    # Покажем несколько файлов
    print(f"\nПервые 3 файла:")
    for i, wav_file in enumerate(wav_files[:3]):
        print(f"  {i+1}: {wav_file}")
    
    # Инициализация модели
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available(): 
        logging.info('GPU доступен')
    else:
        logging.info('Использую CPU')
    
    try:
        import warnings
        warnings.filterwarnings("ignore", message="Unverified HTTPS request")
        
        logging.info("Загрузка модели...")
        wavlm_model = WavLMWrapper.from_pretrained("tiantiaf/wavlm-large-age-sex").to(device)
        wavlm_model.eval()
        logging.info("Модель загружена успешно")
    except Exception as e:
        logging.error(f"Ошибка загрузки модели: {e}")
        exit(1)
    
    # Обработка файлов
    processed_count = 0
    error_count = 0
    
    with open(output_file, 'w', encoding='utf-8') as f_out:
        error_file = "agevoxceleb_errors.log"
        
        for i, wav_path in enumerate(wav_files):
            try:
                if not os.path.exists(wav_path):
                    logging.warning(f"Файл не существует: {wav_path}")
                    error_count += 1
                    continue
                
                audio_tensor = load_voxceleb_audio(wav_path)
                if audio_tensor is None:
                    error_count += 1
                    with open(error_file, 'a') as f_err:
                        f_err.write(f"LOAD_ERROR: {wav_path}\n")
                    continue
                
                audio_tensor = audio_tensor.to(device)
                
                with torch.no_grad():
                    wavlm_age_outputs, wavlm_sex_outputs = wavlm_model(audio_tensor)
                
                age_pred = float(wavlm_age_outputs.detach().cpu().numpy()[0] * 100)
                
                sex_prob = F.softmax(wavlm_sex_outputs, dim=1)
                sex_pred = torch.argmax(sex_prob).detach().cpu().item()
                sex_label = "Male" if sex_pred == 1 else "Female"
                
                # Получаем utterance_id из пути
                rel_path = os.path.relpath(wav_path, voxceleb2_base_path)
                utterance_id = rel_path[:-4] if rel_path.endswith('.wav') else rel_path
                if utterance_id.endswith('.m4a'):
                    utterance_id = utterance_id[:-4]
                
                # Ищем реальный возраст из метаданных
                real_age = None
                for _, row in df_age.iterrows():
                    if row['utterance_id'] in utterance_id:
                        real_age = row['age']
                        break
                
                result_line = f"{utterance_id} {real_age} {age_pred:.2f}"
                f_out.write(result_line + '\n')
                f_out.flush()
                
                processed_count += 1
                
                if (i + 1) % 1000 == 0:
                    logging.info(f"Обработано {i+1}/{len(wav_files)} файлов")
                    
            except Exception as e:
                logging.error(f"Ошибка обработки {wav_path}: {e}")
                error_count += 1
                with open(error_file, 'a') as f_err:
                    f_err.write(f"PROCESS_ERROR: {wav_path} - {str(e)}\n")
                continue
    
    logging.info(f"Готово! Результаты сохранены в {output_file}")
    print(f"\n=== Статистика ===")
    print(f"Всего записей в тесте: {len(utterance_ids)}")
    print(f"Найдено файлов: {len(wav_files)}")
    print(f"Успешно обработано: {processed_count}")
    print(f"Ошибок: {error_count}")
    print(f"Файл с результатами: {output_file}")
    if error_count > 0:
        print(f"Файл с ошибками: agevoxceleb_errors.log")
    
    if os.path.exists(output_file) and processed_count > 0:
        print(f"\nПервые 5 результатов:")
        with open(output_file, 'r') as f:
            for i, line in enumerate(f):
                if i < 5:
                    print(f"  {line.strip()}")
                else:
                    break
