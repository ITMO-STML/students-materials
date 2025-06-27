import streamlit as st
import numpy as np
import soundfile as sf
import librosa
import onnxruntime as ort
import os

# Словарь с информацией о моделях
MODEL_INFO = {
    "(AAR)14-Feb-2025(08-29)_0214_ASV21_DF_MUSAN-epoch=037-eer=0.036.ckpt.onnx": {
        "name": "Rawboost и шумовые аугментации",
        "description": "Обучена на датасете ASVspoof 2021, MUSAN аугментации и включен Rawboost",
        "Equal Error Rate (EER)": "0.1226",
        "Threshold at EER": "0.1300"
    },
    "(WA)14-Feb-2025(09-08)_0214_ASV21_DF_BL-epoch=021-eer=0.040.ckpt.onnx": {
        "name": "Без шумовых аугментаций",
        "description": "Обучена на датасете ASVspoof 2021, без шумовых аугментаций, но включен Rawboost",
        "Equal Error Rate (EER)": "0.1347",
        "Threshold at EER": "0.2765"
    },
    "(WR)28-Mar-2025(11-20)_0214_ASV21_DF_MUSAN-epoch=048-eer=0.041.ckpt.onnx": {
        "name": "Без Rawboost",
        "description": "Обучена на датасете ASVspoof 2021, MUSAN аугментации, но выключен Rawboost",
        "Equal Error Rate (EER)": "0.1140",
        "Threshold at EER": "0.6402"
    },
    "(WRWA)02-Apr-2025(15-17)_0204_ASV21_DF_MUSAN-epoch=024-eer=0.044.ckpt.onnx": {
        "name": "Без шумовых аугментаций и без Rawboost",
        "description": "Обучена на датасете ASVspoof 2021, без аугментаций и выключен Rawboost",
        "Equal Error Rate (EER)": "0.1361",
        "Threshold at EER": "0.5650"
    }
}

# Список доступных моделей (ключи из MODEL_INFO)
MODEL_OPTIONS = list(MODEL_INFO.keys())

# Путь к папке с моделями
MODEL_DIR = "models"

# Функция загрузки модели с кэшированием
@st.cache_resource
def load_model(model_filename):
    model_path = os.path.join(MODEL_DIR, model_filename)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Модель {model_filename} не найдена в директории {MODEL_DIR}")
    return ort.InferenceSession(model_path)

# Интерфейс
st.title("Проверка голоса на спуфинг")
st.write("Выберите модель и загрузите WAV-файл для анализа")

# Выбор модели
selected_model_file = st.selectbox(
    "Выберите модель:",
    options=MODEL_OPTIONS,
    format_func=lambda x: MODEL_INFO[x]["name"]
)

# Отображение информации о выбранной модели
if selected_model_file:
    info = MODEL_INFO[selected_model_file]
    st.markdown(f"### 📦 Информация о модели: {info['name']}")
    st.markdown(f"- **Описание:** {info['description']}")
    st.markdown(f"- **Equal Error Rate (EER):** {info['Equal Error Rate (EER)']}")
    st.markdown(f"- **Порог:** {info['Threshold at EER']}")
    st.divider()

# Загрузка модели
try:
    ort_session = load_model(selected_model_file)
except Exception as e:
    st.error(f"Ошибка загрузки модели: {str(e)}")
    st.stop()

# Предобработка аудио
def preprocess_audio(file_path, target_sr=16000, target_length=64000):
    data, sr = sf.read(file_path)
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    if sr != target_sr:
        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
    if len(data) > target_length:
        data = data[:target_length]
    else:
        data = np.pad(data, (0, target_length - len(data)), 'constant')
    data = data / np.max(np.abs(data))
    return data.astype(np.float32)

# Загрузка файла
uploaded_file = st.file_uploader("Выберите WAV-файл", type=["wav"])

if uploaded_file is not None:
    try:
        file_path = "temp.wav"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        audio_data = preprocess_audio(file_path)
        outputs = ort_session.run(
            None,
            {'waves': audio_data.reshape(1, -1)}
        )
        
        # Обработка результата
        result = "Живая речь" if np.argmax(outputs[0]) == 1 else "Спуфинг"
        confidence = float(np.max(outputs[0]))
        
        st.success(f"Результат: {result}")
        st.info(f"Вероятность: {confidence:.2%}")
        
    except Exception as e:
        st.error(f"Ошибка при анализе файла: {str(e)}")