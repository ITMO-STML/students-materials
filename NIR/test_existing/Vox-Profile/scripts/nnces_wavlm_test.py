import torch
import sys, os
import torch.nn.functional as F
from pathlib import Path
import numpy as np
import ssl

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

def load_nnces_audio(file_path, target_sr=16000, max_samples=240000):
    """Загружает аудио из NNCES (аналог load_timit_audio)"""
    try:
        import librosa
        
        # Загружаем с исходной частотой
        audio, sr = librosa.load(file_path, sr=None, mono=True)
        
        # Ресемплинг если нужно (как в TIMIT)
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        
        # Ограничение длины
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        elif len(audio) < 1600:
            return None
                
        # Нормализация
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        
        return torch.FloatTensor(audio).unsqueeze(0)
    
    except Exception as e:
        logging.error(f"Ошибка загрузки {file_path}: {e}")
        return None

def find_nnces_wav_files(base_path):
    """Находит все WAV файлы в NNCES (аналог find_original_wav_files)"""
    wav_files = []
    for wav_path in Path(base_path).rglob('*.[Ww][Aa][Vv]'):
        wav_files.append(str(wav_path))
    return sorted(wav_files)

if __name__ == '__main__':
    # Настройки для NNCES
    nnces_base_path = "./nnces_split/test"
    output_file = "nnces_age_predictions.txt"
    
    # Поиск файлов
    logging.info(f"Поиск WAV файлов в {nnces_base_path}...")
    wav_files = find_nnces_wav_files(nnces_base_path)
    
    logging.info(f"Найдено {len(wav_files)} WAV файлов")
    
    if len(wav_files) == 0:
        logging.error("Файлы не найдены! Выход.")
        exit(1)
    
    # Покажем несколько файлов
    print(f"\nПервые 10 файлов:")
    for i, wav_file in enumerate(wav_files[:10]):
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
        error_file = "nnces_errors.log"
        
        for i, wav_path in enumerate(wav_files):
            try:
                if not os.path.exists(wav_path):
                    logging.warning(f"Файл не существует: {wav_path}")
                    error_count += 1
                    continue
                
                audio_tensor = load_nnces_audio(wav_path)
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
                
                rel_path = os.path.relpath(wav_path, nnces_base_path)
                result_line = f"{rel_path} {age_pred:.2f}"
            
                f_out.write(result_line + '\n')
                f_out.flush()
                
                processed_count += 1
                
                if (i + 1) % 20 == 0:
                    logging.info(f"Обработано {i+1}/{len(wav_files)} файлов")
                    
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
        print(f"Файл с ошибками: nnces_errors.log")
    
    if os.path.exists(output_file) and processed_count > 0:
        print(f"\nПервые 5 результатов:")
        with open(output_file, 'r') as f:
            for i, line in enumerate(f):
                if i < 5:
                    print(f"  {line.strip()}")
                else:
                    break
