import numpy as np
import sounddevice as sd
import librosa
import soundfile as sf
import torch
import os
import tempfile
from pystoi import stoi
from scipy.signal import resample
from torch import nn
import matplotlib.pyplot as plt
import warnings
import noisereduce as nr
import sys

class Config:
    SAMPLE_RATE = 48000
    TARGET_SR = 16000
    RECORD_DURATION = 10  # Общая длительность записи в секундах
    SEGMENT_DURATION = 3  # Длительность сегмента в секундах
    STEP_SIZE = 1  # Шаг между сегментами в секундах
    N_FFT = 512
    HOP_LENGTH = 256
    N_MELS = 64
    VAD_PARAMS = {
        'top_db': 25,
        'frame_length': 2048,
        'hop_length': 512
    }
    MODEL_PATH = "best_model.pth"  # Путь к модели
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TARGET_FRAMES = 88

warnings.filterwarnings("ignore", category=RuntimeWarning)

class STOIPredictor(nn.Module):
    def __init__(self):
        super(STOIPredictor, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.flatten_size = 5632

        self.fc_layers = nn.Sequential(
            nn.Linear(self.flatten_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)

        if x.size(1) != self.flatten_size:
            x = nn.functional.interpolate(
                x.unsqueeze(0).unsqueeze(0),
                size=self.flatten_size,
                mode='linear'
            ).squeeze(0).squeeze(0)

        x = self.fc_layers(x)
        return x

# Загрузка модели
def load_model():
    model = STOIPredictor().to(Config.DEVICE)
    try:
        state_dict = torch.load(Config.MODEL_PATH, map_location=Config.DEVICE)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        return None

# Запись аудио с микрофона
def record_audio():
    print(f"\nГоворите в микрофон {Config.RECORD_DURATION} секунд...")
    audio = sd.rec(
        int(Config.RECORD_DURATION * Config.SAMPLE_RATE),
        samplerate=Config.SAMPLE_RATE,
        channels=1,
        dtype='float32'
    )
    sd.wait()
    return audio.flatten()

# Нарезка аудио на сегменты
def segment_audio(audio, sr):
    """Нарезает аудио на сегменты с заданным шагом"""
    segment_length = int(Config.SEGMENT_DURATION * sr)
    step_samples = int(Config.STEP_SIZE * sr)
    
    segments = []
    for start in range(0, len(audio) - segment_length + 1, step_samples):
        end = start + segment_length
        segment = audio[start:end]
        segments.append(segment)
    
    return segments

# Ресемплирование аудио
def resample_audio(audio, original_sr, target_sr):
    return librosa.resample(audio, orig_sr=original_sr, target_sr=target_sr)

# VAD
def energy_based_vad(y, sr, top_db=20, frame_length=2048, hop_length=512):
    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
    energy = np.sum(frames**2, axis=0)
    energy_db = librosa.amplitude_to_db(energy, ref=np.max)
    speech_frames = energy_db > -top_db

    speech_audio = []
    for i, is_speech in enumerate(speech_frames):
        if is_speech:
            start = i * hop_length
            end = start + frame_length
            speech_audio.extend(y[start:end])

    return np.array(speech_audio) if len(speech_audio) > 0 else y

# Обработка аудио (VAD и ресемплинг)
def process_audio(audio):
    try:
        audio_vad = resample_audio(audio, Config.SAMPLE_RATE, Config.TARGET_SR)
        speech_only = energy_based_vad(audio_vad, Config.TARGET_SR)
        speech_for_model = resample_audio(speech_only, Config.TARGET_SR, Config.TARGET_SR)

        return {
            'original': audio,
            'speech_only': speech_only,
            'speech_for_model': speech_for_model,
            'sr_original': Config.SAMPLE_RATE,
            'sr_processed': Config.TARGET_SR
        }
    except Exception as e:
        print(f"Ошибка обработки аудио: {e}")
        return None

# Извлечение мел-спектрограммы
def extract_mel_spectrogram(audio, sr):
    try:
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=Config.N_FFT,
            hop_length=Config.HOP_LENGTH,
            n_mels=Config.N_MELS
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        if np.max(mel_spec_db) - np.min(mel_spec_db) > 0:
            mel_spec_db = (mel_spec_db - np.min(mel_spec_db)) / (np.max(mel_spec_db) - np.min(mel_spec_db))
        else:
            mel_spec_db = np.zeros_like(mel_spec_db)

        if mel_spec_db.shape[1] < Config.TARGET_FRAMES:
            pad_width = ((0, 0), (0, Config.TARGET_FRAMES - mel_spec_db.shape[1]))
            mel_spec_db = np.pad(mel_spec_db, pad_width, mode='constant')
        else:
            mel_spec_db = mel_spec_db[:, :Config.TARGET_FRAMES]

        return mel_spec_db
    except Exception as e:
        print(f"Ошибка извлечения мел-спектрограммы: {e}")
        return np.zeros((Config.N_MELS, Config.TARGET_FRAMES))

# Оценка SNR
def estimate_snr(audio, sr=16000, frame_length=2048, hop_length=512):
    if len(audio) < frame_length:
        print("[ERROR] Audio length less than frame length!")
        return 0.0

    noisy_part = audio[:frame_length*3]
    reduced_noise = nr.reduce_noise(y=audio, y_noise=noisy_part, sr=sr)

    signal_energy = np.mean(reduced_noise**2)
    noise_energy = np.mean((audio - reduced_noise)**2)

    snr_linear = signal_energy / (noise_energy + 1e-10)
    snr_db = 10 * np.log10(snr_linear)

    return snr_db

# Предсказание STOI
def predict_stoi(model, audio, sr):
    if sr != Config.TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=Config.TARGET_SR)

    audio = audio / np.max(np.abs(audio))

    n_samples = Config.TARGET_SR * Config.SEGMENT_DURATION
    if len(audio) > n_samples:
        audio = audio[:n_samples]
    else:
        padding = n_samples - len(audio)
        audio = np.pad(audio, (0, padding), mode='constant')

    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=Config.N_FFT,
        hop_length=Config.HOP_LENGTH,
        n_mels=Config.N_MELS
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())

    mel_spec_tensor = torch.FloatTensor(mel_spec_db).unsqueeze(0).to(Config.DEVICE)

    with torch.no_grad():
        prediction =  model(mel_spec_tensor)

    return max(0.0, min(1.0, prediction.item()))

# Сохранение аудиофайлов
def save_audio_segments(audio_dict, temp_dir=None):
    if temp_dir is None:
        temp_dir = tempfile.gettempdir()

    paths = {
        'original': os.path.join(temp_dir, "original.wav"),
        'speech_only': os.path.join(temp_dir, "speech_only.wav")
    }

    sf.write(paths['original'], audio_dict['original'], audio_dict['sr_original'])
    sf.write(paths['speech_only'], audio_dict['speech_only'], audio_dict['sr_processed'])
    return paths

# Визуализация аудиосигналов
def plot_audio_waveforms(audio_dict, save_path=None):
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        ax1.plot(audio_dict['original'])
        ax1.set_title('Original Audio')
        ax1.set_xlabel('Samples')
        ax1.set_ylabel('Amplitude')

        ax2.plot(audio_dict['speech_only'])
        ax2.set_title('Speech Only (after VAD)')
        ax2.set_xlabel('Samples')
        ax2.set_ylabel('Amplitude')

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close()
            return True
        else:
            plt.show()
            return True
    except:
        return False

# Основная функция
def main():
    model = load_model()
    if model is None:
        print("Не удалось загрузить модель. Завершение работы.")
        return

    print(f"Модель загружена на устройство: {Config.DEVICE}")
    print("Доступные аудио устройства:")
    print(sd.query_devices())

    sd.default.device = None
    sd.default.samplerate = Config.SAMPLE_RATE

    while True:
        user_input = input("\nНажмите Enter для начала записи или 'q' для выхода... ")
        
        if user_input.lower() == 'q':
            print("\nЗавершение работы программы...")
            sys.exit(0)
            
        elif user_input != '':
            continue

        try:
            audio = record_audio()
            if len(audio) == 0:
                print("Ошибка: не удалось записать аудио")
                continue

            segments = segment_audio(audio, Config.SAMPLE_RATE)
            if not segments:
                print("Ошибка: не удалось нарезать аудио на сегменты")
                continue

            print(f"\nАудио разделено на {len(segments)} сегментов по {Config.SEGMENT_DURATION} секунд с шагом {Config.STEP_SIZE} сек")

            stoi_values = []
            snr_values = []
            
            for i, segment in enumerate(segments):
                processed = process_audio(segment)
                if processed is None:
                    print(f"Ошибка обработки сегмента {i+1}")
                    continue

                vad_stoi = predict_stoi(model, processed['speech_for_model'], Config.TARGET_SR)
                snr = estimate_snr(processed['speech_for_model'])
                
                stoi_values.append(vad_stoi)
                snr_values.append(snr)
                
                print(f"\nСегмент {i+1} (с {i*Config.STEP_SIZE:.1f} по {i*Config.STEP_SIZE+Config.SEGMENT_DURATION:.1f} сек):")
                print(f"STOI: {vad_stoi:.4f}")
                print(f"SNR: {snr:.2f} dB")
                
                print("Качество речи:", end=" ")
                if vad_stoi > 0.85:
                    print("Отличное")
                elif vad_stoi > 0.7:
                    print("Хорошее")
                elif vad_stoi > 0.5:
                    print("Удовлетворительное")
                elif vad_stoi > 0.3:
                    print("Низкое")
                else:
                    print("Очень плохое")

            if stoi_values:
                avg_stoi = np.mean(stoi_values)
                avg_snr = np.mean(snr_values)
                
                print("\n=== Итоговые результаты ===")
                print(f"Средний STOI: {avg_stoi:.4f}")
                print(f"Средний SNR: {avg_snr:.2f} dB")
                
                print("\nОбщее качество речи:", end=" ")
                if avg_stoi > 0.85:
                    print("Отличное качество!")
                elif avg_stoi > 0.7:
                    print("Хорошее качество")
                elif avg_stoi > 0.5:
                    print("Удовлетворительное качество")
                elif avg_stoi > 0.3:
                    print("Низкое качество")
                else:
                    print("Очень плохое качество")

                plot_path = os.path.join(tempfile.gettempdir(), "audio_waveforms.png")
                plot_success = plot_audio_waveforms(processed, plot_path)
                if plot_success:
                    print(f"\nВизуализация последнего сегмента сохранена: {plot_path}")

        except KeyboardInterrupt:
            print("\nЗавершение работы...")
            break
        except Exception as e:
            print(f"\nОшибка: {str(e)}")

if __name__ == "__main__":
    main()
