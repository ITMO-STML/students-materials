import numpy as np
import librosa
import pandas as pd
import os
import argparse
import logging
from pathlib import Path
from tqdm import tqdm

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s %(levelname)-3s ==> %(message)s', 
    level=logging.INFO, 
    datefmt='%Y-%m-%d %H:%M:%S'
)


# Поиск записей
def find_original_wav_files(timit_base_path):
    """Находит только оригинальные .WAV файлы """
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


def load_timit_audio(file_path, target_sr=16000, max_samples=240000):
    """Загружает аудио из TIMIT"""
    try:
        try:
            from scipy.io import wavfile
            sr, audio = wavfile.read(file_path)
            audio = audio.astype(np.float32)
        except Exception:
            import librosa
            audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
        except Exception:
            import soundfile as sf
            audio, sr = sf.read(file_path, dtype='float32')
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
        
        if sr != target_sr:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        elif len(audio) < 1600:
            logging.warning(f"Аудио слишком короткое: {file_path}")
            return None
        
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        return audio, target_sr
    
    except Exception as e:
        logging.error(f"Ошибка загрузки {file_path}: {e}")
        return None


def extract_mfcc_and_filterbanks(audio, sr=16000, n_mfcc=40, n_mels=40, 
                                 hop_length=512, win_length=1024):
    """Извлекает MFCC и фильтрбанки."""
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=n_mels,
        hop_length=hop_length, win_length=win_length, power=2.0
    )
    filterbanks = librosa.power_to_db(mel_spec, ref=np.max)
    mfcc = librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=n_mfcc, n_mels=n_mels,
        hop_length=hop_length, win_length=win_length,
        dct_type=2, norm='ortho'
    )
    return mfcc.T, filterbanks.T


def parse_spkrinfo(file_path):
    """Парсит SPKRINFO.TXT."""
    spkr_info = {}
    
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith(';') or 'ID  Sex DR Use' in line:
                continue
            if '--' in line or not line.strip():
                continue
            
            parts = line.strip().split()
            if len(parts) >= 9:
                speaker_id = parts[0]
                sex = parts[1]
                use = parts[3]
                rec_date = parts[4]
                birth_date = parts[5]
                
                try:
                    from datetime import datetime
                    rec_dt = datetime.strptime(rec_date, '%m/%d/%y')
                    birth_dt = datetime.strptime(birth_date, '%m/%d/%y')
                    
                    if birth_dt > rec_dt:
                        birth_dt = birth_dt.replace(year=birth_dt.year - 100)
                    
                    age = (rec_dt - birth_dt).days / 365.25
                    
                    spkr_info[speaker_id] = {
                        'age': age,
                        'sex': sex,
                        'split': use
                    }
                except Exception as e:
                    logging.debug(f"Ошибка парсинга для {speaker_id}: {e}")
                    continue
    
    return spkr_info


def get_speaker_id_from_path(file_path):
    """
    Извлекает speaker_id из пути TIMIT.
    Путь: .../data/TRAIN/DR1/FAKS0/SA1.WAV -> AKS0 (отбрасываем первую букву)
    """
    parts = file_path.split(os.sep)
    for part in parts:
        # ID спикера в пути: 5 символов, первый - буква пола (F/M), остальные 4 — ID
        if len(part) == 5 and part[0] in ['F', 'M'] and part[1:].isalnum():
            # Возвращаем без первой буквы (AKS0, SLS0, MDM0)
            return part[1:]  # FAKS0 -> AKS0
        elif len(part) == 4 and part[-1].isdigit() and part[:-1].isalpha():
            return part
    return None


def main():
    # Путь к TIMIT
    TIMIT_PATH = "./darpa-timit-acousticphonetic-continuous-speech/versions/6/"
    OUTPUT_DIR = "./timit_features"
    N_MFCC = 40
    N_MELS = 40
    SR = 16000
    
    # Выходные папки
    os.makedirs(f"{OUTPUT_DIR}/mfcc", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/fbank", exist_ok=True)

    logging.info("\nИзвлечение признаков из TIMIT")
    logging.info(f"Путь к TIMIT: {TIMIT_PATH}")
    logging.info(f"MFCC: {N_MFCC} коэффициентов")
    logging.info(f"Filterbanks: {N_MELS} банков")
    logging.info(f"Частота: {SR} Гц")
    
    # Поиск WAV файлов
    logging.info("\n[1/4] Поиск оригинальных .WAV файлов...")
    wav_files = find_original_wav_files(TIMIT_PATH)
    logging.info(f"Найдено {len(wav_files)} оригинальных .WAV файлов")
    
    if len(wav_files) == 0:
        logging.error("Файлы не найдены!")
        return
    
    # Показываем примеры
    print(f"\nПервые 5 найденных файлов:")
    for i, wav_file in enumerate(wav_files[:5]):
        print(f"  {i+1}: {wav_file}")
    if len(wav_files) > 5:
        print(f"  ... и еще {len(wav_files) - 5} файлов")
    
    # Загрузка метаданных
    logging.info("\n[2/4] Загрузка метаданных...")
    spkrinfo_path = os.path.join(TIMIT_PATH, 'SPKRINFO.TXT')
    
    if os.path.exists(spkrinfo_path):
        spkr_info = parse_spkrinfo(spkrinfo_path)
        logging.info(f"Загружено метаданных для {len(spkr_info)} спикеров")
        
        # Покажем примеры speaker_id из метаданных
        sample_ids = list(spkr_info.keys())[:10]
        logging.info(f"Примеры speaker_id из SPKRINFO: {sample_ids}")
    else:
        logging.error(f"Файл {spkrinfo_path} не найден!")
        return
    
    # Извлечение признаков
    logging.info("\n[3/4] Извлечение признаков...")
    
    all_metadata = []
    processed_count = 0
    error_count = 0
    no_speaker_id_count = 0
    no_metadata_count = 0
    
    # Покажем пример speaker_id из пути для проверки
    logging.info("Проверка извлечения speaker_id из путей:")
    for wav_path in wav_files[:5]:
        sid = get_speaker_id_from_path(wav_path)
        logging.info(f"  {os.path.basename(wav_path)} -> speaker_id: {sid}")
    
    for wav_path in tqdm(wav_files, desc="Обработка файлов"):
        # Извлекаем speaker_id
        speaker_id = get_speaker_id_from_path(wav_path)
        if speaker_id is None:
            no_speaker_id_count += 1
            continue
        
        # Получаем метаданные
        meta = spkr_info.get(speaker_id, {})
        if not meta:
            no_metadata_count += 1
            continue
        
        # Загружаем аудио
        audio_data = load_timit_audio(wav_path, target_sr=SR)
        if audio_data is None:
            error_count += 1
            continue
        
        audio, sr = audio_data
        
        try:
            mfcc, filterbanks = extract_mfcc_and_filterbanks(
                audio, sr=SR, 
                n_mfcc=N_MFCC, n_mels=N_MELS
            )
        except Exception as e:
            logging.error(f"Ошибка извлечения из {wav_path}: {e}")
            error_count += 1
            continue
        
        # Сохраняем
        base_name = os.path.basename(wav_path).replace('.WAV', '')
        mfcc_path = f"{OUTPUT_DIR}/mfcc/{speaker_id}_{base_name}.npy"
        fbank_path = f"{OUTPUT_DIR}/fbank/{speaker_id}_{base_name}.npy"
        
        np.save(mfcc_path, mfcc)
        np.save(fbank_path, filterbanks)
        
        all_metadata.append({
            'speaker_id': speaker_id,
            'file_name': base_name,
            'file_path': wav_path,
            'age': meta['age'],
            'sex': meta['sex'],
            'split': meta['split'],
            'mfcc_shape': mfcc.shape,
            'filterbanks_shape': filterbanks.shape
        })
        
        processed_count += 1
    
    # Сохранение метаданных
    logging.info("\n[4/4] Сохранение метаданных...")
    metadata_df = pd.DataFrame(all_metadata)
    metadata_df.to_csv(f"{OUTPUT_DIR}/features_metadata.csv", index=False)
    
    logging.info(f"\nСтатистика")
    logging.info(f"Всего файлов: {len(wav_files)}")
    logging.info(f"Успешно обработано: {processed_count}")
    logging.info(f"Не удалось извлечь speaker_id: {no_speaker_id_count}")
    logging.info(f"Нет метаданных в SPKRINFO: {no_metadata_count}")
    logging.info(f"Ошибок загрузки/извлечения: {error_count}")
    logging.info(f"\nРезультаты сохранены в:")
    logging.info(f"  - MFCC: {OUTPUT_DIR}/mfcc/")
    logging.info(f"  - Filterbanks: {OUTPUT_DIR}/fbank/")
    logging.info(f"  - Метаданные: {OUTPUT_DIR}/features_metadata.csv")
    
    if processed_count > 0:
        train_count = sum(1 for m in all_metadata if m['split'] == 'TRN')
        test_count = sum(1 for m in all_metadata if m['split'] == 'TST')
        logging.info(f"\nСтатистика по сплитам:")
        logging.info(f"  - TRAIN: {train_count} файлов")
        logging.info(f"  - TEST: {test_count} файлов")
        
        male_count = sum(1 for m in all_metadata if m['sex'] == 'M')
        female_count = sum(1 for m in all_metadata if m['sex'] == 'F')
        logging.info(f"\nСтатистика по полу:")
        logging.info(f"  - Мужчины: {male_count}")
        logging.info(f"  - Женщины: {female_count}")


if __name__ == "__main__":
    main()