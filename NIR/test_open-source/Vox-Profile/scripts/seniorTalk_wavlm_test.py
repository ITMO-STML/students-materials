import torch
import sys, os
import torch.nn.functional as F
from pathlib import Path
import numpy as np
import ssl
from datasets import load_dataset
import logging

ssl._create_default_https_context = ssl._create_unverified_context

sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[1])))
sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[1]), 'model', 'age_sex'))

from model.wavlm_demographics import WavLMWrapper

logging.basicConfig(
    format='%(asctime)s %(levelname)-3s ==> %(message)s', 
    level=logging.INFO, 
    datefmt='%Y-%m-%d %H:%M:%S'
)

os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1"

def load_seniortalk_audio(audio_array, sampling_rate=16000, target_sr=16000, max_samples=240000):
    """Загружает аудио из SeniorTalk"""
    try:
        audio = torch.FloatTensor(audio_array.copy())
        
        if sampling_rate != target_sr:
            import librosa
            audio_np = audio.numpy()
            audio_np = librosa.resample(audio_np, orig_sr=sampling_rate, target_sr=target_sr)
            audio = torch.FloatTensor(audio_np)
        
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        elif len(audio) < 1600:
            return None
                
        audio = audio / (torch.max(torch.abs(audio)) + 1e-8)
        return audio.unsqueeze(0)
    
    except Exception as e:
        logging.error(f"Ошибка аудио: {e}")
        return None

if __name__ == '__main__':
    # Загружаем SeniorTalk вместо TIMIT/NNCES
    output_file = "seniortalk_predictions.txt"

    logging.info("Загрузка SeniorTalk...")
    ds = load_dataset("evan0617/seniortalk", "sentence_data", split="test")
    logging.info(f"Найдено {len(ds)} записей")
    
    # Покажем несколько файлов
    print(f"\nПервые 5 записей:")
    for i in range(min(5, len(ds))):
        filename = ds[i]['path']['path']
        print(f"  {i+1}: {filename}")
    
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
        error_file = "seniortalk_errors.log"
        
        for i in range(len(ds)):
            try:
                # Берем данные из датасета
                audio_array = ds[i]['path']['array']
                sampling_rate = ds[i]['path']['sampling_rate']
                filename = ds[i]['path']['path']
                
                audio_tensor = load_seniortalk_audio(audio_array, sampling_rate)
                if audio_tensor is None:
                    error_count += 1
                    with open(error_file, 'a') as f_err:
                        f_err.write(f"LOAD_ERROR: {filename}\n")
                    continue
                
                audio_tensor = audio_tensor.to(device)
                
                with torch.no_grad():
                    wavlm_age_outputs, wavlm_sex_outputs = wavlm_model(audio_tensor)
                
                age_pred = float(wavlm_age_outputs.detach().cpu().numpy()[0] * 100)
                
                sex_prob = F.softmax(wavlm_sex_outputs, dim=1)
                sex_pred = torch.argmax(sex_prob).detach().cpu().item()
                sex_label = "Male" if sex_pred == 1 else "Female"
                
                result_line = f"{filename} {age_pred:.2f}"
                f_out.write(result_line + '\n')
                f_out.flush()
                
                processed_count += 1
                
                if (i + 1) % 20 == 0:
                    logging.info(f"Обработано {i+1}/{len(ds)} записей")
                    
            except Exception as e:
                logging.error(f"Ошибка {filename}: {e}")
                error_count += 1
                with open(error_file, 'a') as f_err:
                    f_err.write(f"ERROR: {filename} - {str(e)}\n")
                continue
    
    logging.info(f"Готово! Результаты в {output_file}")
    print(f"\n=== Статистика ===")
    print(f"Всего записей: {len(ds)}")
    print(f"Обработано: {processed_count}")
    print(f"Ошибок: {error_count}")
    print(f"Файл результатов: {output_file}")
    if error_count > 0:
        print(f"Файл с ошибками: timit_errors.log")

    if os.path.exists(output_file) and processed_count > 0:
        print(f"\nПервые 5 результатов:")
        with open(output_file, 'r') as f:
            for i, line in enumerate(f):
                if i < 5:
                    print(f"  {line.strip()}")
                else:
                    break
