"""
Демонстрационная программа для показа работы CNN модели предсказания STOI.
Использует Gradio для создания веб-интерфейса.
"""
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import librosa
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Используем non-interactive backend
from pathlib import Path
import gradio as gr

# Добавляем путь к src для импорта модели
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from model import CNNSTOIPredictor

# Константы
SAMPLE_RATE = 16000
CHUNK_DURATION = 5.0  # 5 секунд
CHUNK_STEP = 1.0  # Шаг в 1 секунду
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Путь к модели (можно изменить через переменную окружения)
DEFAULT_CHECKPOINT = '/home/danya/develop/speech_intelligibility_assessment/checkpoints_all_models/best_cnn.pt'
CHECKPOINT_PATH = os.getenv('MODEL_CHECKPOINT', DEFAULT_CHECKPOINT)

# Глобальная переменная для хранения модели (загружается один раз)
_loaded_model = None


def load_model(checkpoint_path):
    """Загружает модель из чекпоинта"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Чекпоинт не найден: {checkpoint_path}")
    
    print(f"Загрузка модели из {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    
    # Получаем параметры модели из чекпоинта
    if 'model_kwargs' in checkpoint:
        model_kwargs = checkpoint['model_kwargs']
    elif 'hyperparameters' in checkpoint and 'model_kwargs' in checkpoint['hyperparameters']:
        model_kwargs = checkpoint['hyperparameters']['model_kwargs']
    else:
        # Используем параметры по умолчанию из лучшей конфигурации
        model_kwargs = {
            'input_dim': 1,
            'num_filters': [96, 192, 384, 768, 1536],
            'kernel_sizes': [11, 9, 7, 5, 3],
            'stride': 3,
            'dropout': 0.06,
            'use_metadata_features': False,
            'fc_hidden_dim': 320,
            'num_fc_layers': 3
        }
    
    # Создаем модель
    model = CNNSTOIPredictor(**model_kwargs).to(DEVICE)
    
    # Загружаем веса
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print(f"Модель загружена успешно. Параметров: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


def split_audio_into_chunks(audio, sample_rate, chunk_duration=5.0, step=1.0):
    """
    Разделяет аудио на чанки заданной длительности с заданным шагом.
    
    Args:
        audio: numpy array с аудио данными
        sample_rate: частота дискретизации
        chunk_duration: длительность чанка в секундах
        step: шаг между чанками в секундах
    
    Returns:
        chunks: список чанков (каждый как numpy array)
        chunk_times: список временных меток начала каждого чанка (в секундах)
    """
    chunk_samples = int(chunk_duration * sample_rate)
    step_samples = int(step * sample_rate)
    
    chunks = []
    chunk_times = []
    
    start_idx = 0
    while start_idx + chunk_samples <= len(audio):
        chunk = audio[start_idx:start_idx + chunk_samples]
        chunks.append(chunk)
        chunk_times.append(start_idx / sample_rate)
        start_idx += step_samples
    
    return chunks, chunk_times


def predict_stoi(model, audio_chunk, sample_rate=SAMPLE_RATE):
    """
    Предсказывает STOI для одного чанка аудио.
    
    Args:
        model: обученная модель
        audio_chunk: numpy array с аудио данными
        sample_rate: частота дискретизации
    
    Returns:
        stoi_pred: предсказанное значение STOI
    """
    # Нормализуем аудио
    if len(audio_chunk) == 0:
        return 0.0
    
    # Ресемплинг до нужной частоты, если необходимо
    if sample_rate != SAMPLE_RATE:
        audio_chunk = librosa.resample(audio_chunk, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
    
    # Конвертируем в tensor
    audio_tensor = torch.FloatTensor(audio_chunk).to(DEVICE)
    
    # Добавляем batch dimension
    audio_tensor = audio_tensor.unsqueeze(0)  # (1, seq_len)
    
    # Предсказание
    with torch.no_grad():
        stoi_pred = model(audio_tensor)
        stoi_pred = stoi_pred.cpu().item()
    
    return float(stoi_pred)


def process_audio(audio_file):
    """
    Обрабатывает аудио файл и возвращает результаты.
    
    Args:
        audio_file: путь к аудио файлу или tuple (sample_rate, audio_data)
    
    Returns:
        tuple: (график STOI, среднее значение, мел-спектр с наложенными значениями)
    """
    if audio_file is None:
        return None, "Загрузите аудио файл", None
    
    try:
        # Загружаем аудио
        if isinstance(audio_file, tuple):
            # Gradio передает (sample_rate, audio_data)
            sample_rate, audio_data = audio_file
            if audio_data.ndim > 1:
                # Если стерео, берем первый канал
                audio = audio_data[:, 0].astype(np.float32)
            else:
                audio = audio_data.astype(np.float32)
        else:
            # Это путь к файлу
            audio, sample_rate = librosa.load(audio_file, sr=None, mono=True)
        
        # Проверяем длину аудио
        duration = len(audio) / sample_rate
        if duration < CHUNK_DURATION:
            return None, f"Аудио слишком короткое ({duration:.2f} сек). Нужно минимум {CHUNK_DURATION} секунд.", None
        
        # Нормализуем
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        
        # Разделяем на чанки
        chunks, chunk_times = split_audio_into_chunks(
            audio, sample_rate, CHUNK_DURATION, CHUNK_STEP
        )
        
        if len(chunks) == 0:
            return None, "Аудио слишком короткое (нужно минимум 5 секунд)", None
        
        # Загружаем модель (делаем это один раз при первом вызове)
        global _loaded_model
        if _loaded_model is None:
            _loaded_model = load_model(CHECKPOINT_PATH)
        
        # Предсказываем STOI для каждого чанка
        stoi_predictions = []
        for chunk in chunks:
            stoi = predict_stoi(_loaded_model, chunk, sample_rate)
            stoi_predictions.append(stoi)
        
        stoi_predictions = np.array(stoi_predictions)
        chunk_times = np.array(chunk_times)
        
        # Вычисляем среднее значение
        mean_stoi = np.mean(stoi_predictions)
        
        # Создаем график STOI
        fig_stoi, ax_stoi = plt.subplots(figsize=(10, 6))
        ax_stoi.plot(chunk_times, stoi_predictions, 'b-o', linewidth=2, markersize=8)
        ax_stoi.axhline(y=mean_stoi, color='r', linestyle='--', linewidth=2, label=f'Среднее: {mean_stoi:.3f}')
        ax_stoi.set_xlabel('Время (секунды)', fontsize=12)
        ax_stoi.set_ylabel('STOI', fontsize=12)
        ax_stoi.set_title('Предсказанные значения STOI по времени', fontsize=14, fontweight='bold')
        ax_stoi.grid(True, alpha=0.3)
        ax_stoi.legend(fontsize=11)
        ax_stoi.set_ylim([0, 1])
        plt.tight_layout()
        
        # Создаем мел-спектр с наложенными значениями STOI
        # Используем весь аудио сигнал для мел-спектра
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sample_rate, n_mels=128, hop_length=512
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        fig_mel, ax_mel = plt.subplots(figsize=(12, 6))
        
        # Отображаем мел-спектр
        times_mel = librosa.frames_to_time(np.arange(mel_spec_db.shape[1]), sr=sample_rate, hop_length=512)
        im = ax_mel.imshow(mel_spec_db, aspect='auto', origin='lower', 
                          extent=[times_mel[0], times_mel[-1], 0, 128],
                          cmap='viridis', interpolation='bilinear')
        
        # Накладываем значения STOI
        # Для каждого чанка рисуем прямоугольник с цветом, соответствующим значению STOI
        for i, (chunk_time, stoi_val) in enumerate(zip(chunk_times, stoi_predictions)):
            chunk_end = chunk_time + CHUNK_DURATION
            # Используем цветовую карту для отображения STOI (зеленый = высокий, красный = низкий)
            # Нормализуем STOI для цветовой карты (STOI в диапазоне 0-1)
            color = plt.cm.RdYlGn(stoi_val)  # Red-Yellow-Green colormap
            # Рисуем полупрозрачный прямоугольник
            rect = plt.Rectangle((chunk_time, 0), CHUNK_DURATION, 128, 
                               facecolor=color, alpha=0.4, edgecolor='white', linewidth=1.5)
            ax_mel.add_patch(rect)
            # Добавляем текст с значением STOI (только если чанк достаточно большой)
            if CHUNK_DURATION >= 2.0:
                text_color = 'white' if stoi_val < 0.5 else 'black'
                ax_mel.text(chunk_time + CHUNK_DURATION / 2, 64, f'{stoi_val:.2f}',
                           ha='center', va='center', fontsize=9, fontweight='bold',
                           color=text_color, bbox=dict(boxstyle='round,pad=0.3', 
                           facecolor='white', alpha=0.7, edgecolor='black', linewidth=0.5))
        
        # Добавляем цветовую шкалу для STOI
        from matplotlib.colors import LinearSegmentedColormap
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar2 = plt.colorbar(sm, ax=ax_mel, label='STOI', location='right', pad=0.02)
        cbar2.set_label('STOI (наложено на спектр)', rotation=270, labelpad=20)
        
        ax_mel.set_xlabel('Время (секунды)', fontsize=12)
        ax_mel.set_ylabel('Частота (мел-бин)', fontsize=12)
        ax_mel.set_title('Мел-спектр с наложенными значениями STOI', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax_mel, label='Амплитуда (dB)')
        plt.tight_layout()
        
        # Формируем текст с результатами
        result_text = f"""
**Результаты анализа:**

- **Количество чанков:** {len(chunks)}
- **Длительность каждого чанка:** {CHUNK_DURATION} секунд
- **Шаг между чанками:** {CHUNK_STEP} секунд
- **Среднее значение STOI:** {mean_stoi:.4f}
- **Минимальное значение STOI:** {np.min(stoi_predictions):.4f}
- **Максимальное значение STOI:** {np.max(stoi_predictions):.4f}
- **Стандартное отклонение:** {np.std(stoi_predictions):.4f}
        """
        
        return fig_stoi, result_text, fig_mel
        
    except Exception as e:
        import traceback
        error_msg = f"Ошибка при обработке аудио: {str(e)}\n\n{traceback.format_exc()}"
        return None, error_msg, None


# Создаем интерфейс Gradio
def create_interface():
    """Создает интерфейс Gradio"""
    
    with gr.Blocks(title="STOI Prediction Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎤 Демонстрация модели предсказания STOI 🎤
        
        Эта программа позволяет оценить разборчивость речи (STOI - Speech Transmission Objective Intelligibility)
        с помощью обученной CNN модели.
        
        **Инструкция:**
        1. Нажмите кнопку "Записать аудио" и произнесите несколько секунд речи
        2. Или загрузите аудио файл (WAV, MP3 и т.д.)
        3. Аудио будет автоматически разделено на 5-секундные отрезки с шагом 1 секунда
        4. Для каждого отрезка модель предскажет значение STOI
        5. Результаты отобразятся на графиках ниже
        """)
        
        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(
                    label="Записать или загрузить аудио",
                    type="numpy",
                    sources=["microphone", "upload"]
                )
                process_btn = gr.Button("Обработать аудио", variant="primary", size="lg")
            
            with gr.Column():
                results_text = gr.Markdown(label="Результаты")
        
        with gr.Row():
            stoi_plot = gr.Plot(label="График значений STOI")
            mel_spectrogram = gr.Plot(label="Мел-спектр с наложенными значениями STOI")
        
        # Обработка при нажатии кнопки или загрузке аудио
        process_btn.click(
            fn=process_audio,
            inputs=audio_input,
            outputs=[stoi_plot, results_text, mel_spectrogram]
        )
        
        audio_input.change(
            fn=process_audio,
            inputs=audio_input,
            outputs=[stoi_plot, results_text, mel_spectrogram]
        )
        
        # gr.Markdown("""
        # ---
        # **Примечание:** Модель работает лучше всего с чистой речью без сильных шумов.
        # Рекомендуется записывать в тихой обстановке с хорошим микрофоном.
        # """)
    
    return demo


if __name__ == "__main__":
    # Проверяем наличие модели
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"⚠️  Внимание: Чекпоинт не найден по пути: {CHECKPOINT_PATH}")
        print("Установите переменную окружения MODEL_CHECKPOINT или поместите модель в указанный путь.")
        print("Пример: export MODEL_CHECKPOINT=../checkpoints_cnn_final/best_cnn_final.pt")
    
    # Создаем и запускаем интерфейс
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",  # Доступно извне
        server_port=7860,
        share=False  # Установите True для создания публичной ссылки
    )
