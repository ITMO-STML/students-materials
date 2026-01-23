import torch
import sys, os, pdb
import argparse, logging
import torch.nn.functional as F
from pathlib import Path
import numpy as np
import ssl

# Отключаем проверку SSL (для корпоративных прокси)
ssl._create_default_https_context = ssl._create_unverified_context

sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[1])))
sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[1]), 'model', 'age_sex'))

from model.age_sex.wavlm_demographics import WavLMWrapper

import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-3s ==> %(message)s', 
    level=logging.INFO, 
    datefmt='%Y-%m-%d %H:%M:%S'
)

os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

def load_timit_audio(file_path, target_sr=16000, max_samples=240000):
    """Загружает аудио из TIMIT (формат NIST SPHERE)"""
    try:
        try:
            from scipy.io import wavfile
            sr, audio = wavfile.read(file_path)
            audio = audio.astype(np.float32)
        except Exception as e1:
            import librosa
            audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
        except Exception as e2:
            import soundfile as sf
            audio, sr = sf.read(file_path, dtype='float32')
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
        
        # Ресемплируем если нужно
        if sr != target_sr:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        
        # Ограничиваем до 15 секунд (16000 Гц * 15 сек = 240000 отсчетов)
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        elif len(audio) < 1600:  # Минимум 0.1 секунды
            logging.warning(f"Аудио слишком короткое: {file_path} ({len(audio)/sr:.2f} сек)")
            return None
        
        # Нормализуем
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        
        return torch.FloatTensor(audio).unsqueeze(0)
    
    except Exception as e:
        logging.error(f"Ошибка загрузки {file_path}: {e}")
        return None

def filter_unique_wav_files(wav_files):
    """Фильтрует дубликаты и оставляет только оригинальные .WAV файлы"""
    unique_files = []
    seen_bases = set()
    
    for file_path in wav_files:
        base_name = os.path.basename(file_path)
        dir_name = os.path.dirname(file_path)
        
        if base_name.lower().endswith('.wav.wav'):
            original_name = base_name[:-4]  
            file_key = (dir_name, original_name.upper())
        else:
            file_key = (dir_name, base_name.upper())
        
        if file_key not in seen_bases:
            seen_bases.add(file_key)
            if base_name.endswith('.WAV'):
                unique_files.append(file_path)
    
    return sorted(unique_files)

def find_original_wav_files(timit_base_path):
    """Находит только оригинальные .WAV файлы (без дубликатов .wav.wav)"""
    import os
    
    wav_files = []
    
    def _search(path):
        try:
            for item in os.listdir(path):
                item_path = os.path.join(path, item)
                if os.path.isdir(item_path):
                    _search(item_path)
                # Берем только файлы с расширением .WAV (в верхнем регистре)
                elif item.endswith('.WAV'):
                    wav_files.append(item_path)
        except Exception as e:
            logging.debug(f"Не удалось прочитать {path}: {e}")
    
    if os.path.exists(timit_base_path):
        _search(timit_base_path)
    
    return sorted(wav_files)

if __name__ == '__main__':
    # Настройки
    timit_base_path = "./darpa-timit-acousticphonetic-continuous-speech/versions/6"
    output_file = "timit_age_predictions.txt"
    
    logging.info(f"Поиск оригинальных .WAV файлов в {timit_base_path}...")
    wav_files = find_original_wav_files(timit_base_path)
    
    logging.info(f"Найдено {len(wav_files)} оригинальных .WAV файлов")
    
    if len(wav_files) == 0:
        logging.error("Файлы не найдены! Выход.")
        exit(1)
    
    # Покажем несколько найденных файлов для проверки
    print(f"\nПервые 10 найденных файлов:")
    for i, wav_file in enumerate(wav_files[:10]):
        print(f"  {i+1}: {wav_file}")
    
    if len(wav_files) > 10:
        print(f"  ... и еще {len(wav_files) - 10} файлов")
    
    # Инициализация модели
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available(): 
        logging.info('GPU доступен, использую GPU')
    else:
        logging.info('Использую CPU')
    
    try:
        import warnings
        warnings.filterwarnings("ignore", message="Unverified HTTPS request")
        
        logging.info("Загрузка модели... (может занять время)")
        wavlm_model = WavLMWrapper.from_pretrained("tiantiaf/wavlm-large-age-sex").to(device)
        wavlm_model.eval()
        logging.info("Модель загружена успешно")
    except Exception as e:
        logging.error(f"Ошибка загрузки модели: {e}")
        exit(1)
    
    processed_count = 0
    error_count = 0
    
    # Создаем файл для результатов
    with open(output_file, 'w', encoding='utf-8') as f_out:
        error_file = "timit_errors.log"
        
        for i, wav_path in enumerate(wav_files):
            try:
                if not os.path.exists(wav_path):
                    logging.warning(f"Файл не существует: {wav_path}")
                    error_count += 1
                    continue
                
                audio_tensor = load_timit_audio(wav_path)
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
                
                # Сохраняем результат
                rel_path = os.path.relpath(wav_path, timit_base_path)
                result_line = f"{rel_path} {age_pred:.2f}"
                f_out.write(result_line + '\n')
                f_out.flush()  
                
                processed_count += 1
                
                if (i + 1) % 20 == 0:
                    logging.info(f"Обработано {i+1}/{len(wav_files)} файлов (успешно: {processed_count}, ошибок: {error_count})")
                    
            except Exception as e:
                logging.error(f"Ошибка обработки {wav_path}: {e}")
                error_count += 1
                with open(error_file, 'a') as f_err:
                    f_err.write(f"PROCESS_ERROR: {wav_path} - {str(e)}\n")
                continue
    
    logging.info(f"Готово! Результаты сохранены в {output_file}")
    print(f"\n=== Статистика ===")
    print(f"Всего файлов: {len(wav_files)}")
    print(f"Успешно обработано: {processed_count}")
    print(f"Ошибок: {error_count}")
    print(f"Файл с результатами: {output_file}")
    if error_count > 0:
        print(f"Файл с ошибками: timit_errors.log")
    
    # Показываем первые несколько результатов
    if os.path.exists(output_file) and processed_count > 0:
        print(f"\nПервые 5 результатов:")
        with open(output_file, 'r') as f:
            for i, line in enumerate(f):
                if i < 5:
                    print(f"  {line.strip()}")
                else:
                    break