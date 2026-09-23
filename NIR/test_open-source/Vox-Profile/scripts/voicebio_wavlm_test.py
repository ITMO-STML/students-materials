import torch
import sys, os
import torch.nn.functional as F
from pathlib import Path
import numpy as np
import pandas as pd
import ssl
import logging
from tqdm import tqdm

ssl._create_default_https_context = ssl._create_unverified_context

sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[1])))
sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[1]), 'model', 'age_sex'))

from model.age_sex.wavlm_demographics import WavLMWrapper

logging.basicConfig(format='%(asctime)s %(levelname)-3s ==> %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

def load_audio(file_path, target_sr=16000, max_samples=240000):
    try:
        import librosa
        audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        elif len(audio) < 1600:
            return None
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        return torch.FloatTensor(audio).unsqueeze(0)
    except Exception as e:
        logging.error(f"Ошибка загрузки {file_path}: {e}")
        return None

if __name__ == '__main__':
    dataset_path = "./voice_biometrics_age/voice_biometrics_age"
    excel_path = "./voice_biometrics_age/metadata.xlsx"
    output_csv = "voicebio_predictions.csv"
    
    df = pd.read_excel(excel_path, engine='openpyxl')
    df['date of birth'] = pd.to_datetime(df['date of birth'], errors='coerce')
    
    id_to_birth_year = {}
    for idx, row in df.iterrows():
        if pd.notna(row['date of birth']):
            id_to_birth_year[row['id']] = row['date of birth'].year
    
    logging.info(f"Загружено {len(id_to_birth_year)} персон с известной датой рождения")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f'Использую {device}')
    
    wavlm_model = WavLMWrapper.from_pretrained("tiantiaf/wavlm-large-age-sex").to(device)
    wavlm_model.eval()
    
    results = []
    
    for person_dir in tqdm(list(Path(dataset_path).iterdir()), desc="Обработка персон"):
        if not person_dir.is_dir() or not person_dir.name.startswith('id'):
            continue
        
        person_id = person_dir.name
        
        if person_id not in id_to_birth_year:
            logging.warning(f"Нет даты рождения для {person_id}, пропуск")
            continue
        
        birth_year = id_to_birth_year[person_id]
        
        for year_dir in person_dir.iterdir():
            if not year_dir.is_dir():
                continue
            
            try:
                record_year = int(year_dir.name)
                age = record_year - birth_year
                
                # Проверяем разумный возраст
                if age < 0 or age > 100:
                    continue
                
                # Ищем WAV файлы
                wav_files = list(year_dir.glob('*.wav'))
                
                for audio_path in wav_files:
                    try:
                        audio_tensor = load_audio(str(audio_path))
                        if audio_tensor is None:
                            continue
                            
                        audio_tensor = audio_tensor.to(device)
                        
                        with torch.no_grad():
                            age_output, _ = wavlm_model(audio_tensor)
                        
                        pred_age = float(age_output.detach().cpu().numpy()[0] * 100)
                        
                        results.append({
                            'id': person_id,
                            'year': record_year,
                            'true_age': age,
                            'pred_age': round(pred_age, 2),
                            'file': str(audio_path)
                        })
                        
                    except Exception as e:
                        logging.error(f"Ошибка обработки {audio_path}: {e}")
                        continue
                        
            except ValueError:
                continue
    
    # Сохраняем в CSV
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_csv, index=False)
    
    logging.info(f"Сохранено {len(results)} записей в {output_csv}")
    
    print(f"\nСтатистика")
    print(f"Всего людей в мета: {len(id_to_birth_year)}")
    print(f"Найдено персон в датасете: {df_results['id'].nunique()}")
    print(f"Обработано файлов: {len(results)}")
    print(f"\nПервые 5 записей:")
    print(df_results.head())
    
    print(f"\nРаспределение:")
    print(df_results['id'].value_counts().head(10))